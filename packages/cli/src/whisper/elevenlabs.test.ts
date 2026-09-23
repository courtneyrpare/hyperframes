import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  ELEVENLABS_DEFAULT_STT_MODEL,
  ELEVENLABS_STT_ENDPOINT,
  resolveElevenLabsModel,
  scribeWordsToWords,
  transcribeWithElevenLabs,
} from "./elevenlabs.js";

const SCRIBE_RESPONSE = {
  language_code: "en",
  text: "Hello world",
  words: [
    { text: "Hello", start: 0.12, end: 0.5, type: "word" },
    { text: " ", start: 0.5, end: 0.55, type: "spacing" },
    { text: "(laughter)", start: 0.55, end: 0.9, type: "audio_event" },
    { text: "world", start: 0.9, end: 1.3, type: "word" },
  ],
};

describe("scribeWordsToWords", () => {
  it("keeps spoken words and drops spacing and audio events", () => {
    expect(scribeWordsToWords(SCRIBE_RESPONSE)).toEqual([
      { text: "Hello", start: 0.12, end: 0.5 },
      { text: "world", start: 0.9, end: 1.3 },
    ]);
  });
});

describe("resolveElevenLabsModel", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("prefers the explicit model, then the env override, then the default", () => {
    vi.stubEnv("HYPERFRAMES_ELEVENLABS_STT_MODEL", "");
    expect(resolveElevenLabsModel()).toBe(ELEVENLABS_DEFAULT_STT_MODEL);
    vi.stubEnv("HYPERFRAMES_ELEVENLABS_STT_MODEL", "scribe_env");
    expect(resolveElevenLabsModel()).toBe("scribe_env");
    expect(resolveElevenLabsModel("scribe_flag")).toBe("scribe_flag");
  });
});

describe("transcribeWithElevenLabs", () => {
  let dir: string;
  let input: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hf-elevenlabs-test-"));
    input = join(dir, "narration.wav");
    writeFileSync(input, "not-real-audio");
    vi.stubEnv("ELEVENLABS_API_KEY", "test-key");
    vi.stubEnv("HYPERFRAMES_ELEVENLABS_STT_MODEL", "");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    rmSync(dir, { recursive: true, force: true });
  });

  it("posts the file with word timestamps and writes transcript.json", async () => {
    const fetchImpl = vi.fn<typeof fetch>(
      async () => new Response(JSON.stringify(SCRIBE_RESPONSE), { status: 200 }),
    );

    const result = await transcribeWithElevenLabs(input, dir, { language: "en", fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0]!;
    expect(url).toBe(ELEVENLABS_STT_ENDPOINT);
    expect(init?.headers).toEqual({ "xi-api-key": "test-key" });
    const form = init?.body;
    expect(form).toBeInstanceOf(FormData);
    if (!(form instanceof FormData)) return;
    expect(form.get("model_id")).toBe(ELEVENLABS_DEFAULT_STT_MODEL);
    expect(form.get("timestamps_granularity")).toBe("word");
    expect(form.get("tag_audio_events")).toBe("false");
    expect(form.get("language_code")).toBe("en");
    expect(form.get("file")).toBeInstanceOf(Blob);

    expect(result.wordCount).toBe(2);
    expect(result.durationSeconds).toBe(1.3);
    expect(result.speechOnsetSeconds).toBeNull();
    expect(JSON.parse(readFileSync(result.transcriptPath, "utf-8"))).toEqual([
      { text: "Hello", start: 0.12, end: 0.5 },
      { text: "world", start: 0.9, end: 1.3 },
    ]);
  });

  it("surfaces the API error detail", async () => {
    const fetchImpl = vi.fn<typeof fetch>(
      async () =>
        new Response(JSON.stringify({ detail: { message: "Invalid API key" } }), { status: 401 }),
    );
    await expect(transcribeWithElevenLabs(input, dir, { fetchImpl })).rejects.toThrow(
      /HTTP 401\): Invalid API key Check ELEVENLABS_API_KEY/,
    );
  });

  it("fails fast without an API key", async () => {
    vi.stubEnv("ELEVENLABS_API_KEY", "");
    const fetchImpl = vi.fn<typeof fetch>();
    await expect(transcribeWithElevenLabs(input, dir, { fetchImpl })).rejects.toThrow(
      /ELEVENLABS_API_KEY is not set/,
    );
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
