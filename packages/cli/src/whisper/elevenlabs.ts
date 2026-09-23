/**
 * ElevenLabs Scribe transcription engine (cloud, via the Speech-to-Text API).
 *
 * The default engine whenever `ELEVENLABS_API_KEY` is set: `auto` resolves to
 * it ahead of the local Parakeet / whisper.cpp engines. It needs no local
 * model or toolchain, covers 99 languages, and returns word-level timestamps
 * directly, so no token merging is needed. The input file is uploaded to
 * ElevenLabs as-is (the API accepts both audio and video containers).
 */

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { writeWordsTranscript } from "./parakeet.js";
import type { Word } from "./normalize.js";
import type { TranscribeResult } from "./transcribe.js";

export const ELEVENLABS_STT_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text";
export const ELEVENLABS_DEFAULT_STT_MODEL = "scribe_v1";
const DEFAULT_TIMEOUT_MS = 1_800_000;

/** The API key, or undefined when ElevenLabs is not configured. */
export function getElevenLabsApiKey(): string | undefined {
  const key = process.env["ELEVENLABS_API_KEY"]?.trim();
  return key ? key : undefined;
}

/** Scribe model id: explicit option, then `HYPERFRAMES_ELEVENLABS_STT_MODEL`, then the default. */
export function resolveElevenLabsModel(model?: string): string {
  return model || process.env["HYPERFRAMES_ELEVENLABS_STT_MODEL"] || ELEVENLABS_DEFAULT_STT_MODEL;
}

interface ScribeWord {
  text?: unknown;
  start?: unknown;
  end?: unknown;
  type?: unknown;
}
interface ScribeResponse {
  words?: ScribeWord[];
}

function isScribeResponse(value: unknown): value is ScribeResponse {
  if (typeof value !== "object" || value === null) return false;
  return !("words" in value) || value.words === undefined || Array.isArray(value.words);
}

/**
 * Keep only spoken words. Scribe interleaves `spacing` entries (the whitespace
 * between words) and `audio_event` entries such as "(laughter)"; neither
 * belongs in a caption transcript.
 */
export function scribeWordsToWords(response: ScribeResponse): Word[] {
  const words: Word[] = [];
  for (const w of response.words ?? []) {
    if (w.type !== undefined && w.type !== "word") continue;
    if (typeof w.text !== "string" || typeof w.start !== "number") continue;
    const text = w.text.trim();
    if (!text) continue;
    const end = typeof w.end === "number" ? w.end : w.start;
    words.push({ text, start: w.start, end });
  }
  return words;
}

/** Pull the human-readable message out of an API error body (`{ detail: string | { message } }`). */
function extractErrorDetail(body: string): string {
  try {
    const parsed: unknown = JSON.parse(body);
    if (typeof parsed !== "object" || parsed === null || !("detail" in parsed)) return body.trim();
    const d = parsed.detail;
    if (typeof d === "string") return d;
    if (typeof d === "object" && d !== null && "message" in d) return String(d.message);
    return JSON.stringify(d);
  } catch {
    return body.trim();
  }
}

const STATUS_HINTS: Record<number, string> = {
  401: " Check ELEVENLABS_API_KEY.",
  429: " Rate limited; retry.",
};

function describeApiError(status: number, body: string): string {
  const detail = extractErrorDetail(body).slice(0, 500);
  return `ElevenLabs speech-to-text failed (HTTP ${status}): ${detail}${STATUS_HINTS[status] ?? ""}`;
}

interface ElevenLabsOptions {
  language?: string;
  model?: string;
  timeoutMs?: number;
  onProgress?: (message: string) => void;
  /** Injected in tests; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
}

/** Transcribe with ElevenLabs Scribe and write `transcript.json` (Word[]) into `dir`. */
export async function transcribeWithElevenLabs(
  inputPath: string,
  dir: string,
  options?: ElevenLabsOptions,
): Promise<TranscribeResult> {
  const apiKey = getElevenLabsApiKey();
  if (!apiKey) {
    throw new Error(
      "ELEVENLABS_API_KEY is not set. Export your ElevenLabs API key to use the ElevenLabs engine (or use --engine parakeet|whisper).",
    );
  }

  const form = new FormData();
  form.append("model_id", resolveElevenLabsModel(options?.model));
  form.append("timestamps_granularity", "word");
  form.append("tag_audio_events", "false");
  if (options?.language) form.append("language_code", options.language);
  form.append("file", new Blob([readFileSync(inputPath)]), basename(inputPath));

  options?.onProgress?.("Uploading to ElevenLabs...");
  const doFetch = options?.fetchImpl ?? fetch;
  const response = await doFetch(ELEVENLABS_STT_ENDPOINT, {
    method: "POST",
    headers: { "xi-api-key": apiKey },
    body: form,
    signal: AbortSignal.timeout(options?.timeoutMs ?? DEFAULT_TIMEOUT_MS),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(describeApiError(response.status, body));

  const parsed: unknown = JSON.parse(body);
  if (!isScribeResponse(parsed)) throw new Error("ElevenLabs returned an unexpected response.");
  return writeWordsTranscript(dir, scribeWordsToWords(parsed));
}
