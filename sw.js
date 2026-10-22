// Service worker : toute l'appli est mise en cache à l'installation pour marcher hors ligne.
// Changer VERSION à chaque mise en ligne (et dans index.html, vérifié par tests/pwa.test.js),
// sinon les téléphones gardent l'ancienne version.
const VERSION = '0.1.0';
const CACHE = `orientation-voiture-${VERSION}`;

const FICHIERS = [
  './',
  'index.html',
  'style.css',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/apple-touch-icon.png',
  'vendor/tf.min.js',
  'js/main.js',
  'js/camera.js',
  'js/classifieur.js',
  'js/detector.js',
  'js/geometry.js',
  'js/orientation.js',
  'js/render.js',
  'models/yolo11n/model.json',
  'models/yolo11n/group1-shard1of3.bin',
  'models/yolo11n/group1-shard2of3.bin',
  'models/yolo11n/group1-shard3of3.bin',
  'models/orientation/model.json',
  'models/orientation/group1-shard1of1.bin',
];

self.addEventListener('install', (ev) => {
  ev.waitUntil(caches.open(CACHE).then((c) => c.addAll(FICHIERS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (ev) => {
  ev.waitUntil(
    caches.keys()
      .then((noms) => Promise.all(noms.filter((n) => n !== CACHE).map((n) => caches.delete(n))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (ev) => {
  if (ev.request.method !== 'GET') return;
  ev.respondWith(
    caches.match(ev.request, { ignoreSearch: true }).then((r) => r || fetch(ev.request)),
  );
});
