// Caméra arrière dans un <video> caché.

export async function demarrerCamera(video) {
  const flux = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  video.srcObject = flux;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  return video;
}

export function arreterCamera(video) {
  if (video.srcObject) video.srcObject.getTracks().forEach((piste) => piste.stop());
  video.srcObject = null;
}
