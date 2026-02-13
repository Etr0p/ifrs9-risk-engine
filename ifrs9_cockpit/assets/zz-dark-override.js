/**
 * Dash 4.0 Dark Theme Override
 *
 * Dash 4.0 injects light-theme CSS variables (:root { --Dash-Fill-Inverse-Strong: #fff })
 * via its component JS bundles. This script uses MutationObserver to re-apply
 * dark theme tokens every time Dash injects a new <style> tag.
 *
 * Placed in assets/ with zz- prefix to load last.
 */
(function() {
    'use strict';

    var DARK_CSS = [
        ':root {',
        '  --Dash-Fill-Inverse-Strong: #1E293B;',
        '  --Dash-Text-Strong: #F8FAFC;',
        '  --Dash-Text-Primary: #F8FAFC;',
        '  --Dash-Text-Weak: #A1B2C8;',
        '  --Dash-Text-Disabled: #A1B2C8;',
        '  --Dash-Fill-Interactive-Strong: #3B82F6;',
        '  --Dash-Fill-Interactive-Weak: rgba(59,130,246,0.15);',
        '  --Dash-Fill-Disabled: rgba(99,102,241,0.12);',
        '  --Dash-Fill-Primary-Active: rgba(59,130,246,0.15);',
        '  --Dash-Fill-Primary-Hover: rgba(59,130,246,0.08);',
        '  --Dash-Stroke-Strong: rgba(99,102,241,0.25);',
        '  --Dash-Stroke-Weak: rgba(99,102,241,0.12);',
        '  --Dash-Shading-Strong: rgba(0,0,0,0.5);',
        '  --Dash-Shading-Weak: rgba(0,0,0,0.3);',
        '}',
    ].join('\n');

    function applyDarkTokens() {
        var el = document.getElementById('dash-dark-tokens');
        if (!el) {
            el = document.createElement('style');
            el.id = 'dash-dark-tokens';
            document.head.appendChild(el);
        }
        el.textContent = DARK_CSS;
        // Always move to end of <head> to override everything
        document.head.appendChild(el);
    }

    // Apply immediately
    applyDarkTokens();

    // Re-apply whenever Dash injects new <style> tags
    var observer = new MutationObserver(function(mutations) {
        for (var i = 0; i < mutations.length; i++) {
            var addedNodes = mutations[i].addedNodes;
            for (var j = 0; j < addedNodes.length; j++) {
                var node = addedNodes[j];
                if (node.tagName === 'STYLE' && node.id !== 'dash-dark-tokens') {
                    applyDarkTokens();
                    return;
                }
            }
        }
    });
    observer.observe(document.head, { childList: true });

    // Safety net: re-apply after delays (component lazy loading)
    setTimeout(applyDarkTokens, 500);
    setTimeout(applyDarkTokens, 1500);
    setTimeout(applyDarkTokens, 3000);
    setTimeout(applyDarkTokens, 5000);
})();
