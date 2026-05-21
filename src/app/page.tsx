/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";

type Status =
  | "Init"
  | "Loading models"
  | "Starting camera"
  | "Calibrating"
  | "Running"
  | "Stopped"
  | "Error";

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 420;
const VIDEO_FPS = 12;

const DETECT_INTERVAL = 900; // ms
const TARGET_OBJECTS = ["book", "cell phone", "laptop", "remote"];

type FaceMeshLike = {
  setOptions: (opts: any) => void;
  onResults: (cb: (res: any) => void) => void;
  send: (args: { image: HTMLVideoElement }) => Promise<void>;
  initialize?: () => Promise<void>;
};

const median = (arr: number[]) => {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

export default function ProctoringPOCPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const faceMeshRef = useRef<FaceMeshLike | null>(null);
  const cocoRef = useRef<cocoSsd.ObjectDetection | null>(null);

  const rafRef = useRef<number | null>(null);
  const detectTimerRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<Status>("Init");
  const [modelReady, setModelReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  const logIdRef = useRef(0);
  const [log, setLog] = useState<Array<{ id: number; ts: number; text: string }>>([]);
  const pushLog = (text: string) =>
    setLog((prev) =>
      [{ id: ++logIdRef.current, ts: Date.now(), text }, ...prev].slice(0, 10),
    );

  // UI state mirrors of the refs below — updated periodically so the UI repaints.
  const [uiTelemetry, setUiTelemetry] = useState({
    faceCount: 0,
    isLookingAtScreen: true,
    violationReason: null as string | null,
    iris: { leftX: 0.5, rightX: 0.5, leftY: 0.5, rightY: 0.5 },
    baseline: { leftX: 0.5, rightX: 0.5, leftY: 0.5, rightY: 0.5 },
  });

  const [thresholds, setThresholds] = useState({
    horizontal: 0.18,
    vertical: 0.22,
  });

  // Calibration + attention
  const isCalibratingRef = useRef(false);

  const baselineRef = useRef({
    leftX: 0.5,
    rightX: 0.5,
    leftY: 0.5,
    rightY: 0.5,
  });

  const thresholdsRef = useRef(thresholds);
  // Keep the ref in sync with state so the FaceMesh callback always reads the latest value.
  thresholdsRef.current = thresholds;

  const calibrationSamplesRef = useRef({
    leftX: [] as number[],
    rightX: [] as number[],
    leftY: [] as number[],
    rightY: [] as number[],
  });

  const lastFaceInfoRef = useRef({
    faceCount: 0,
    isLookingAtScreen: true,
    violationReason: null as string | null,
    ts: Date.now(),
    iris: { leftX: 0.5, rightX: 0.5, leftY: 0.5, rightY: 0.5 },
  });

  const blobToBase64 = (blob: Blob) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(String(reader.result));
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const captureFrameBase64 = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    canvas.width = VIDEO_WIDTH;
    canvas.height = VIDEO_HEIGHT;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.86),
    );
    if (!blob) return null;

    return blobToBase64(blob);
  };

  const stoppedRef = useRef(false);

  const teardownResources = () => {
    stoppedRef.current = true;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    if (detectTimerRef.current) window.clearTimeout(detectTimerRef.current);
    detectTimerRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  const stopAll = () => {
    teardownResources();
    setCameraReady(false);
    setStatus("Stopped");
    pushLog("Stopped.");
  };

  const setupFaceMesh = async () => {
    // Turbopack-friendly dynamic import
    const mod = await import("@mediapipe/face_mesh");
    const FaceMeshCtor =
      mod.FaceMesh ||
      mod.default?.FaceMesh ||
      mod.default ||
      (window as any).FaceMesh;

    if (!FaceMeshCtor) throw new Error("FaceMesh constructor not found");

    const fm: FaceMeshLike = new FaceMeshCtor({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });

    fm.setOptions({
      refineLandmarks: true, // REQUIRED for iris landmarks 468/473
      maxNumFaces: 2,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    fm.onResults((results: any) => {
      const faces = results?.multiFaceLandmarks ?? [];
      lastFaceInfoRef.current.faceCount = faces.length;
      lastFaceInfoRef.current.ts = Date.now();

      if (faces.length !== 1) {
        lastFaceInfoRef.current.isLookingAtScreen = false;
        lastFaceInfoRef.current.violationReason =
          faces.length === 0 ? "No face detected" : "Multiple faces detected";
        return;
      }

      const face = faces[0];

      const leftIris = face[468];
      const rightIris = face[473];

      const leftEyeInner = face[133];
      const leftEyeOuter = face[33];
      const rightEyeInner = face[362];
      const rightEyeOuter = face[263];

      const leftEyeTop = face[159];
      const leftEyeBottom = face[145];
      const rightEyeTop = face[386];
      const rightEyeBottom = face[374];

      const leftEyeWidth = Math.abs(leftEyeOuter.x - leftEyeInner.x) || 1e-6;
      const rightEyeWidth = Math.abs(rightEyeOuter.x - rightEyeInner.x) || 1e-6;

      const leftEyeHeight = Math.abs(leftEyeTop.y - leftEyeBottom.y) || 1e-6;
      const rightEyeHeight = Math.abs(rightEyeTop.y - rightEyeBottom.y) || 1e-6;

      const leftIrisX = clamp01((leftIris.x - leftEyeInner.x) / leftEyeWidth);
      const rightIrisX = clamp01(
        (rightIris.x - rightEyeInner.x) / rightEyeWidth,
      );

      const leftIrisY = clamp01((leftIris.y - leftEyeTop.y) / leftEyeHeight);
      const rightIrisY = clamp01(
        (rightIris.y - rightEyeTop.y) / rightEyeHeight,
      );

      lastFaceInfoRef.current.iris = {
        leftX: leftIrisX,
        rightX: rightIrisX,
        leftY: leftIrisY,
        rightY: rightIrisY,
      };

      if (isCalibratingRef.current) {
        calibrationSamplesRef.current.leftX.push(leftIrisX);
        calibrationSamplesRef.current.rightX.push(rightIrisX);
        calibrationSamplesRef.current.leftY.push(leftIrisY);
        calibrationSamplesRef.current.rightY.push(rightIrisY);

        if (calibrationSamplesRef.current.leftX.length >= 30) {
          baselineRef.current = {
            leftX: median(calibrationSamplesRef.current.leftX),
            rightX: median(calibrationSamplesRef.current.rightX),
            leftY: median(calibrationSamplesRef.current.leftY),
            rightY: median(calibrationSamplesRef.current.rightY),
          };

          isCalibratingRef.current = false;
          setStatus("Running");
          pushLog(
            `Calibration done. Baselines: LX ${baselineRef.current.leftX.toFixed(
              2,
            )} RX ${baselineRef.current.rightX.toFixed(
              2,
            )} LY ${baselineRef.current.leftY.toFixed(
              2,
            )} RY ${baselineRef.current.rightY.toFixed(2)}`,
          );
        }

        lastFaceInfoRef.current.isLookingAtScreen = true;
        lastFaceInfoRef.current.violationReason = null;
        return;
      }

      const dxL = Math.abs(leftIrisX - baselineRef.current.leftX);
      const dxR = Math.abs(rightIrisX - baselineRef.current.rightX);
      const dyL = Math.abs(leftIrisY - baselineRef.current.leftY);
      const dyR = Math.abs(rightIrisY - baselineRef.current.rightY);

      const lookingAway =
        dxL > thresholdsRef.current.horizontal ||
        dxR > thresholdsRef.current.horizontal ||
        dyL > thresholdsRef.current.vertical ||
        dyR > thresholdsRef.current.vertical;

      lastFaceInfoRef.current.isLookingAtScreen = !lookingAway;
      lastFaceInfoRef.current.violationReason = lookingAway
        ? "Looking away from screen"
        : null;
    });

    if (typeof fm.initialize === "function") await fm.initialize();
    faceMeshRef.current = fm;
  };

  const startCamera = async () => {
    setStatus("Starting camera");

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: VIDEO_WIDTH },
        height: { ideal: VIDEO_HEIGHT },
        frameRate: { ideal: VIDEO_FPS },
      },
      audio: false,
    });

    streamRef.current = stream;

    const video = videoRef.current;
    if (!video) throw new Error("videoRef not ready");

    video.srcObject = stream;

    await new Promise<void>((resolve) => {
      if (video.readyState >= 1) return resolve();
      video.addEventListener("loadedmetadata", () => resolve(), { once: true });
    });

    await video.play();
    setCameraReady(true);
  };

  const faceLoop = async () => {
    const video = videoRef.current;
    const fm = faceMeshRef.current;

    if (video && fm && video.readyState >= 2) {
      try {
        await fm.send({ image: video });
      } catch (e) {
        console.error("FaceMesh send error", e);
      }
    }

    rafRef.current = requestAnimationFrame(faceLoop);
  };

  const startDetectionLoop = () => {
    const tick = async () => {
      if (stoppedRef.current) return;

      const model = cocoRef.current;
      const video = videoRef.current;
      if (!model || !video || video.readyState < 2) {
        detectTimerRef.current = window.setTimeout(tick, DETECT_INTERVAL);
        return;
      }

      try {
        const preds = await model.detect(video);
        const relevant = preds.filter(
          (p) => TARGET_OBJECTS.includes(p.class) && (p.score ?? 0) > 0.6,
        );

        const attentionIssue =
          !lastFaceInfoRef.current.isLookingAtScreen ||
          lastFaceInfoRef.current.faceCount !== 1;

        if (relevant.length > 0 || attentionIssue) {
          // Only build the base64 frame when there's a real forbidden-object hit;
          // it's expensive and not used for plain attention events.
          const imageBase64 =
            relevant.length > 0 ? await captureFrameBase64() : null;

          const payload = {
            ts: Date.now(),
            attention: {
              faceCount: lastFaceInfoRef.current.faceCount,
              ok: lastFaceInfoRef.current.isLookingAtScreen,
              reason: lastFaceInfoRef.current.violationReason,
              iris: lastFaceInfoRef.current.iris,
              baseline: baselineRef.current,
              thresholds: thresholdsRef.current,
            },
            objects: relevant.map((p) => ({
              class: p.class,
              score: Number(p.score ?? 0),
            })),
            imageBase64,
          };

          console.log("POC payload:", payload);

          if (relevant.length > 0) {
            pushLog(
              `Forbidden object: ${relevant
                .map((r) => `${r.class} (${(r.score ?? 0).toFixed(2)})`)
                .join(", ")}`,
            );
          }
          if (attentionIssue) {
            pushLog(`Attention: ${payload.attention.reason ?? "Issue"}`);
          }
        }
      } catch (e) {
        console.error("Detect error", e);
      }

      if (!stoppedRef.current) {
        detectTimerRef.current = window.setTimeout(tick, DETECT_INTERVAL);
      }
    };

    tick();
  };

  const startCalibration = () => {
    calibrationSamplesRef.current = {
      leftX: [],
      rightX: [],
      leftY: [],
      rightY: [],
    };
    isCalibratingRef.current = true;
    setStatus("Calibrating");
    pushLog("Calibration started: look at the center and keep still…");
  };

  const start = async () => {
    try {
      stoppedRef.current = false;
      setStatus("Loading models");
      cocoRef.current = await cocoSsd.load({ base: "lite_mobilenet_v2" });
      await setupFaceMesh();
      setModelReady(true);

      await startCamera();
      setStatus("Running");

      pushLog("Models loaded. Camera started. Click “Calibrate gaze”.");

      rafRef.current = requestAnimationFrame(faceLoop);
      startDetectionLoop();
    } catch (e) {
      console.error(e);
      setStatus("Error");
      pushLog("Failed to start. Check console.");
      stopAll();
    }
  };

  useEffect(() => {
    start();
    return () => teardownResources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Pump the mutable refs into React state at ~5Hz so the telemetry panel repaints.
  useEffect(() => {
    const id = window.setInterval(() => {
      setUiTelemetry({
        faceCount: lastFaceInfoRef.current.faceCount,
        isLookingAtScreen: lastFaceInfoRef.current.isLookingAtScreen,
        violationReason: lastFaceInfoRef.current.violationReason,
        iris: { ...lastFaceInfoRef.current.iris },
        baseline: { ...baselineRef.current },
      });
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  const badge = useMemo(() => {
    if (status === "Running") return "Live";
    if (status === "Calibrating") return "Calibrating";
    if (status === "Error") return "Error";
    if (status === "Stopped") return "Stopped";
    return "Starting";
  }, [status]);

  const attentionOk =
    uiTelemetry.faceCount === 1 && uiTelemetry.isLookingAtScreen;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* background */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(900px_600px_at_20%_10%,rgba(99,102,241,0.28),transparent_55%),radial-gradient(800px_520px_at_85%_20%,rgba(34,197,94,0.18),transparent_55%),radial-gradient(900px_600px_at_50%_95%,rgba(244,63,94,0.14),transparent_60%)]" />
        <div className="absolute inset-0 bg-linear-to-b from-slate-950 via-slate-950 to-black" />
      </div>

      <div className="mx-auto max-w-6xl px-5 py-6">
        {/* header */}
        <header className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-5 py-4 shadow-[0_20px_60px_rgba(0,0,0,0.35)] backdrop-blur">
          <div className="flex items-center gap-3">
            <span className="h-3 w-3 rounded-full bg-linear-to-br from-sky-400 to-violet-400 shadow-[0_0_0_6px_rgba(99,102,241,0.15)]" />
            <div>
              <div className="text-base font-extrabold tracking-tight">
                Proctoring POC
              </div>
              <div className="text-xs text-slate-300/80">
                Next.js • FaceMesh (iris calibration) • coco-ssd
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={[
                "rounded-full border px-3 py-1 text-xs font-bold tracking-wide",
                status === "Running"
                  ? "border-emerald-400/30 bg-emerald-500/15 text-emerald-100"
                  : status === "Calibrating"
                    ? "border-indigo-400/35 bg-indigo-500/15 text-indigo-100"
                    : status === "Error"
                      ? "border-rose-400/35 bg-rose-500/15 text-rose-100"
                      : "border-white/15 bg-white/5 text-slate-200/90",
              ].join(" ")}
            >
              {badge}
            </span>
          </div>
        </header>

        <main className="mt-5 grid gap-5 lg:grid-cols-[1.25fr_0.85fr]">
          {/* Left: Preview */}
          <section className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-[0_18px_55px_rgba(0,0,0,0.35)] backdrop-blur">
            <div className="flex items-baseline justify-between">
              <div>
                <div className="text-sm font-extrabold">Camera Preview</div>
                <div className="mt-0.5 text-xs text-slate-300/80">
                  {cameraReady ? "Camera ready" : "Starting…"} •{" "}
                  {modelReady ? "Models ready" : "Loading…"}
                </div>
              </div>

              <div className="text-xs text-slate-300/80">
                Face count:{" "}
                <span className="font-bold text-slate-100">
                  {uiTelemetry.faceCount}
                </span>
              </div>
            </div>

            <div className="relative mt-3 overflow-hidden rounded-2xl border border-white/10 bg-black/20">
              <video
                ref={videoRef}
                width={VIDEO_WIDTH}
                height={VIDEO_HEIGHT}
                className="h-auto w-full"
                playsInline
                muted
              />

              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full border border-white/15 bg-black/40 px-3 py-1.5 backdrop-blur">
                <span
                  className={[
                    "h-2.5 w-2.5 rounded-full",
                    attentionOk ? "bg-emerald-400" : "bg-rose-400",
                    "shadow-[0_0_0_6px_rgba(255,255,255,0.06)]",
                  ].join(" ")}
                />
                <span className="text-xs font-bold">
                  {attentionOk
                    ? "Attention OK"
                    : (uiTelemetry.violationReason ?? "Attention issue")}
                </span>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <button
                onClick={startCalibration}
                disabled={status !== "Running"}
                className="rounded-xl border border-indigo-400/35 bg-linear-to-b from-indigo-500/35 to-indigo-500/15 px-4 py-2 text-xs font-extrabold tracking-wide text-white shadow-[0_10px_25px_rgba(0,0,0,0.25)] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Calibrate gaze
              </button>

              <button
                onClick={stopAll}
                disabled={status !== "Running" && status !== "Calibrating"}
                className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-xs font-extrabold tracking-wide text-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Stop
              </button>

              <div className="min-w-70 flex-1 rounded-2xl border border-white/10 bg-black/20 p-3">
                <div className="text-xs font-extrabold text-slate-100/90">
                  Sensitivity
                </div>
                <div className="mt-2 grid gap-3 sm:grid-cols-2">
                  <label className="grid grid-cols-[86px_1fr_44px] items-center gap-2 text-xs">
                    <span className="font-bold text-slate-200/80">
                      Horizontal
                    </span>
                    <input
                      type="range"
                      min={0.08}
                      max={0.3}
                      step={0.01}
                      value={thresholds.horizontal}
                      onChange={(e) =>
                        setThresholds((t) => ({
                          ...t,
                          horizontal: Number(e.target.value),
                        }))
                      }
                      className="w-full accent-indigo-400"
                    />
                    <span className="text-right tabular-nums text-slate-200/80">
                      {thresholds.horizontal.toFixed(2)}
                    </span>
                  </label>

                  <label className="grid grid-cols-[86px_1fr_44px] items-center gap-2 text-xs">
                    <span className="font-bold text-slate-200/80">
                      Vertical
                    </span>
                    <input
                      type="range"
                      min={0.1}
                      max={0.35}
                      step={0.01}
                      value={thresholds.vertical}
                      onChange={(e) =>
                        setThresholds((t) => ({
                          ...t,
                          vertical: Number(e.target.value),
                        }))
                      }
                      className="w-full accent-indigo-400"
                    />
                    <span className="text-right tabular-nums text-slate-200/80">
                      {thresholds.vertical.toFixed(2)}
                    </span>
                  </label>
                </div>
              </div>
            </div>

            <canvas ref={canvasRef} className="hidden" />
          </section>

          {/* Right: Telemetry + Logs */}
          <section className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-[0_18px_55px_rgba(0,0,0,0.35)] backdrop-blur">
            <div className="flex items-baseline justify-between">
              <div>
                <div className="text-sm font-extrabold">Telemetry</div>
                <div className="mt-0.5 text-xs text-slate-300/80">
                  Looking:{" "}
                  <span className="font-bold text-slate-100">
                    {uiTelemetry.isLookingAtScreen ? "Yes" : "No"}
                  </span>
                </div>
              </div>
              <div className="text-xs text-slate-300/80">
                {status === "Calibrating" ? "Collecting samples…" : "Ready"}
              </div>
            </div>

            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                <div className="text-[11px] font-extrabold tracking-wide text-slate-200/70">
                  Iris (L)
                </div>
                <div className="mt-1 text-xs font-bold tabular-nums text-slate-100">
                  X {uiTelemetry.iris.leftX.toFixed(2)} • Y{" "}
                  {uiTelemetry.iris.leftY.toFixed(2)}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                <div className="text-[11px] font-extrabold tracking-wide text-slate-200/70">
                  Iris (R)
                </div>
                <div className="mt-1 text-xs font-bold tabular-nums text-slate-100">
                  X {uiTelemetry.iris.rightX.toFixed(2)} • Y{" "}
                  {uiTelemetry.iris.rightY.toFixed(2)}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                <div className="text-[11px] font-extrabold tracking-wide text-slate-200/70">
                  Baseline (L)
                </div>
                <div className="mt-1 text-xs font-bold tabular-nums text-slate-100">
                  X {uiTelemetry.baseline.leftX.toFixed(2)} • Y{" "}
                  {uiTelemetry.baseline.leftY.toFixed(2)}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                <div className="text-[11px] font-extrabold tracking-wide text-slate-200/70">
                  Baseline (R)
                </div>
                <div className="mt-1 text-xs font-bold tabular-nums text-slate-100">
                  X {uiTelemetry.baseline.rightX.toFixed(2)} • Y{" "}
                  {uiTelemetry.baseline.rightY.toFixed(2)}
                </div>
              </div>
            </div>

            <div className="my-4 h-px w-full bg-white/10" />

            <div className="text-sm font-extrabold">Recent events</div>
            <div className="mt-3 space-y-3 max-h-60 overflow-auto">
              {log.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-white/15 bg-black/10 p-3 text-xs text-slate-200/70">
                  No events yet.
                </div>
              ) : (
                log.map((l) => (
                  <div
                    key={l.id}
                    className="flex gap-3 rounded-2xl border border-white/10 bg-black/20 p-3"
                  >
                    <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-indigo-400 shadow-[0_0_0_6px_rgba(99,102,241,0.12)]" />
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-slate-100">
                        {l.text}
                      </div>
                      <div className="mt-1 text-[11px] text-slate-200/60">
                        {new Date(l.ts).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-linear-to-b from-indigo-500/10 to-black/10 p-3 text-xs text-slate-100/85">
              Tip: Click <span className="font-extrabold">Calibrate gaze</span>{" "}
              while staring at the center. If it’s too strict, increase
              thresholds.
            </div>
          </section>
        </main>

        
      </div>
    </div>
  );
}
