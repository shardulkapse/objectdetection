# Object Detection

Real-time webcam object detection in the browser. Runs entirely client-side — no backend, no API calls.

**Live demo:** [objectdetection-six.vercel.app](https://objectdetection-six.vercel.app)


## What it does

Grants webcam access in the browser, then streams the live video through a COCO-SSD model running on TensorFlow.js. Detected objects are drawn as labelled bounding boxes overlaid on the video in real time.

Everything runs in the browser. The webcam feed never leaves the device, and there's no backend cost.

## Features

- Real-time object detection from the device webcam
- 4 object classes (the COCO dataset categories COCO-SSD ships with)
- Bounding boxes with class labels and confidence scores
- Built with Next.js App Router and TypeScript

## Tech stack

- **Next.js 15** (App Router)
- **TypeScript**
- **TensorFlow.js** with `@tensorflow-models/coco-ssd`
- **shadcn/ui** + **Tailwind CSS**

## Running locally

```bash
git clone https://github.com/shardulkapse/objectdetection.git
cd objectdetection
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and allow webcam access when prompted.

