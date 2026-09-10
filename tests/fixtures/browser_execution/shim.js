/**
 * Minimal static JS shim for Asupersync browser execution.
 *
 * Implements microtask scheduling, host-turn (macrotask) yielding via MessageChannel,
 * timer delays, non-reentrant pump invocation, and structured event collection.
 * Strictly no eval, no dynamic script injection.
 */
'use strict';

class BrowserExecutionShim {
    constructor(options = {}) {
        this.burstLimit = options.burstLimit || 32;
        this.inPump = false;
        this.macrotaskId = 0;
        this.events = [];
        this.onEvent = options.onEvent || null;
        this.channel = new MessageChannel();
        this.channel.port1.onmessage = () => this._onMacroTurn();
        this.scheduledMacro = false;
        this.pendingMacroCallbacks = [];
    }

    /**
     * Schedules a microtask using the standard host queueMicrotask API.
     */
    scheduleMicrotask(fn) {
        if (typeof queueMicrotask === 'function') {
            queueMicrotask(fn);
        } else {
            Promise.resolve().then(fn);
        }
    }

    /**
     * Yields control to the browser macrotask event loop via MessageChannel.
     */
    yieldHostTurn(callback) {
        this.pendingMacroCallbacks.push(callback);
        if (!this.scheduledMacro) {
            this.scheduledMacro = true;
            this.channel.port2.postMessage(undefined);
        }
    }

    _onMacroTurn() {
        this.scheduledMacro = false;
        this.macrotaskId++;
        const callbacks = this.pendingMacroCallbacks;
        this.pendingMacroCallbacks = [];
        for (const cb of callbacks) {
            try {
                cb(this.macrotaskId);
            } catch (err) {
                console.error('[f3d-shim] Error in macrotask callback:', err);
            }
        }
    }

    /**
     * Schedules a timer delay using standard host setTimeout.
     */
    scheduleTimer(delayMs, callback) {
        return setTimeout(() => {
            this.macrotaskId++;
            callback(this.macrotaskId);
        }, delayMs);
    }

    /**
     * Invokes the pump with non-reentrancy protection.
     * Returns true if executed, false if reentrant call was prevented.
     */
    pumpStep(pumpFn) {
        if (this.inPump) {
            this.recordEvent({
                probe: 'reentrancy_guard',
                task_id: 0,
                event: 'reentrancy_prevented',
                source: 'microtask',
                macrotask_id: this.macrotaskId,
                ts_wall_ms: performance.now(),
                detail: 'Synchronous pump reentrancy rejected by JS shim guard'
            });
            return false;
        }

        this.inPump = true;
        try {
            return pumpFn();
        } finally {
            this.inPump = false;
        }
    }

    /**
     * Records a structured telemetry event.
     */
    recordEvent(event) {
        if (!event.ts_wall_ms) {
            event.ts_wall_ms = Math.round(performance.now());
        }
        if (!event.macrotask_id) {
            event.macrotask_id = this.macrotaskId;
        }
        this.events.push(event);
        if (this.onEvent) {
            this.onEvent(event);
        }
    }

    /**
     * Exports all events as NDJSON / JSON Lines.
     */
    exportNdjson() {
        return this.events.map(e => JSON.stringify(e)).join('\n');
    }
}

if (typeof window !== 'undefined') {
    window.BrowserExecutionShim = BrowserExecutionShim;
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BrowserExecutionShim };
}
