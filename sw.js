/* S&P 500 Oracle service worker: offline shell; data always fetched fresh when online. */
var SHELL='oracle-shell-v3';
self.addEventListener('install',function(e){e.waitUntil(caches.open(SHELL).then(function(c){return c.addAll(['./','./index.html','./icon.svg']);}).catch(function(){}));self.skipWaiting();});
self.addEventListener('activate',function(e){e.waitUntil(caches.keys().then(function(ks){return Promise.all(ks.filter(function(k){return k!==SHELL;}).map(function(k){return caches.delete(k);}));}).then(function(){return self.clients.claim();}));});
self.addEventListener('fetch',function(e){var u=new URL(e.request.url);if(e.request.method!=='GET'||u.origin!==self.location.origin)return;
 e.respondWith(fetch(e.request).then(function(r){if(r.ok){var cp=r.clone();caches.open(SHELL).then(function(c){c.put(e.request,cp);});}return r;}).catch(function(){return caches.match(e.request,{ignoreSearch:true}).then(function(r){return r||(e.request.mode==='navigate'?caches.match('./'):Response.error());});}));});
