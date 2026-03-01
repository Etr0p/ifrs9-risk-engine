/**
 * Connection Watchdog — auto-reload on stale server connection.
 *
 * Watches the watchdog-store dcc.Store for heartbeat updates.
 * If no update is received within 90 seconds, assumes the server
 * connection is dead (e.g. after use_reloader restart, WebSocket timeout,
 * or process crash) and forces a page reload.
 *
 * This prevents the "frozen page" symptom where the browser tab is alive
 * but can no longer communicate with the Dash server.
 */
(function() {
    'use strict';

    var STALE_THRESHOLD_MS = 90000;  // 90s sans heartbeat = reload
    var CHECK_INTERVAL_MS = 15000;   // Verifier toutes les 15s
    var _lastHeartbeat = Date.now();
    var _started = false;

    function checkStale() {
        var elapsed = Date.now() - _lastHeartbeat;
        if (elapsed > STALE_THRESHOLD_MS && _started) {
            console.warn('[WATCHDOG] Connection stale (' + Math.round(elapsed/1000) + 's). Reloading...');
            window.location.reload();
        }
    }

    // Observer le store watchdog pour detecter les heartbeats
    // Le store Dash met a jour un element hidden avec id="watchdog-store"
    var storeObserver = new MutationObserver(function() {
        _lastHeartbeat = Date.now();
        _started = true;
    });

    // Attendre que le DOM Dash soit pret
    function init() {
        var store = document.getElementById('watchdog-store');
        if (store) {
            storeObserver.observe(store, { attributes: true, childList: true, subtree: true });
            _lastHeartbeat = Date.now();
            _started = true;
            setInterval(checkStale, CHECK_INTERVAL_MS);
        } else {
            // Store pas encore rendu, reessayer
            setTimeout(init, 2000);
        }
    }

    // Demarrer apres le chargement initial (laisser le temps au pipeline initial)
    setTimeout(init, 10000);
})();
