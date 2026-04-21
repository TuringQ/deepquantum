/**
 * iframe 嵌入时：全屏不改变 iframe 内布局，由父窗口 document 进入全屏。
 *
 * 为何不能只用 postMessage：postMessage 在父子窗口间多为异步投递，父页面里再调
 * requestFullscreen 时已丢失「用户激活」，浏览器会拒绝首次全屏，表现为常要点第二下。
 *
 * 策略：与父页同源时，在按钮点击的同一同步调用栈内对 window.top.documentElement
 * 调用 requestFullscreen（用户激活仍有效）。仅跨域时才走 postMessage（父页见 parent-page.vue）。
 * 另监听父 document 的 fullscreenchange，与 Esc 退出等情况对齐 expectParentFullscreen。
 *
 * 全屏按钮气泡：已进入全屏时提示「Halfscreen mode」（下一步为退出），否则为「Fullscreen mode」。
 */
(function () {
  var MSG_ENTER = 'deepquantum-docs:request-browser-fullscreen';
  var MSG_EXIT = 'deepquantum-docs:exit-browser-fullscreen';

  var TIP_FULLSCREEN = 'Fullscreen mode';
  var TIP_HALFSCREEN = 'Halfscreen mode';

  var expectParentFullscreen = false;

  function isEmbeddedInFrame() {
    try {
      return window.self !== window.top;
    } catch (e) {
      return true;
    }
  }

  function syncExpectFromTopDocument() {
    try {
      if (window.top === window.self) return;
      var d = window.top.document;
      expectParentFullscreen = !!(d.fullscreenElement || d.webkitFullscreenElement);
    } catch (e) {}
  }

  function notifyParentEnter() {
    try {
      window.parent.postMessage({ type: MSG_ENTER }, '*');
      expectParentFullscreen = true;
    } catch (e) {}
  }

  function notifyParentExit() {
    try {
      window.parent.postMessage({ type: MSG_EXIT }, '*');
    } catch (e) {}
    expectParentFullscreen = false;
  }

  function getFullscreenButton() {
    return document.querySelector('.btn-fullscreen-button');
  }

  /** 与 PyData TriggerTooltip 相同配置，避免重建后行为不一致 */
  var TOOLTIP_OPTS = { delay: { show: 500, hide: 100 } };

  /**
   * 主题在 DOMContentLoaded 上会用 TriggerTooltip 给带 data-bs-toggle 的元素建 Tooltip。
   * 若我们先于主题 new Tooltip，会被第二次初始化搞乱；故 install 里用 setTimeout(0) 晚于主题再同步。
   * BS5 会把文案挪到 data-bs-original-title，只改 title 不一定更新已存在的实例。
   */
  function applyFullscreenButtonTooltip(isFullscreen) {
    var btn = getFullscreenButton();
    if (!btn) return;
    var text = isFullscreen ? TIP_HALFSCREEN : TIP_FULLSCREEN;
    btn.setAttribute('title', text);
    btn.setAttribute('aria-label', text);
    btn.setAttribute('data-bs-original-title', text);
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
      var inst = bootstrap.Tooltip.getInstance(btn);
      if (inst) inst.dispose();
      try {
        new bootstrap.Tooltip(btn, TOOLTIP_OPTS);
      } catch (e) {}
    }
  }

  /** 独立页看本 document；iframe 内优先读同源 top.document，跨域则用 expectParentFullscreen */
  function isBrowserFullscreenEffective() {
    if (isEmbeddedInFrame()) {
      syncExpectFromTopDocument();
      try {
        if (window.top !== window.self) {
          var d = window.top.document;
          return !!(d.fullscreenElement || d.webkitFullscreenElement);
        }
      } catch (e) {}
      return expectParentFullscreen;
    }
    return !!(document.fullscreenElement || document.webkitFullscreenElement);
  }

  function refreshFullscreenButtonTooltip() {
    applyFullscreenButtonTooltip(isBrowserFullscreenEffective());
  }

  /**
   * 同源父页面：与点击同同步栈内请求全屏。
   * @returns {boolean} true 表示已发起原生请求（含返回 Promise），false 表示应回退 postMessage
   */
  function trySyncTopEnterFromSameOriginClick() {
    try {
      if (window.top === window.self) return false;
      var el = window.top.document.documentElement;
      var req = null;
      if (el.requestFullscreen) {
        try {
          req = el.requestFullscreen({ navigationUI: 'hide' });
        } catch (e1) {
          req = el.requestFullscreen();
        }
      }
      if (req !== undefined && req !== null && typeof req.then === 'function') {
        req
          .then(function () {
            expectParentFullscreen = true;
            refreshFullscreenButtonTooltip();
          })
          .catch(function () {
            notifyParentEnter();
            refreshFullscreenButtonTooltip();
          });
        return true;
      }
      if (el.webkitRequestFullscreen) {
        el.webkitRequestFullscreen();
        expectParentFullscreen = true;
        return true;
      }
    } catch (e) {
      return false;
    }
    return false;
  }

  function trySyncTopExitFromSameOrigin() {
    try {
      if (window.top === window.self) return false;
      var d = window.top.document;
      var fs = d.fullscreenElement || d.webkitFullscreenElement;
      if (!fs) return false;
      var p = d.exitFullscreen ? d.exitFullscreen() : null;
      if (p !== undefined && p !== null && typeof p.then === 'function') {
        void p.catch(function () {});
      } else if (d.webkitExitFullscreen) {
        d.webkitExitFullscreen();
      } else {
        return false;
      }
      return true;
    } catch (e) {
      return false;
    }
  }

  function toggleEmbeddedFullscreen() {
    syncExpectFromTopDocument();

    if (expectParentFullscreen) {
      if (trySyncTopExitFromSameOrigin()) {
        expectParentFullscreen = false;
      } else {
        notifyParentExit();
      }
      refreshFullscreenButtonTooltip();
      return;
    }

    if (!trySyncTopEnterFromSameOriginClick()) {
      notifyParentEnter();
    }
    refreshFullscreenButtonTooltip();
  }

  function attachTopFullscreenListeners() {
    try {
      if (window.top === window.self) return;
      var d = window.top.document;
      var onFs = function () {
        syncExpectFromTopDocument();
        refreshFullscreenButtonTooltip();
      };
      d.addEventListener('fullscreenchange', onFs);
      d.addEventListener('webkitfullscreenchange', onFs);
    } catch (e) {}
  }

  function bindStandaloneFullscreenTooltipListeners() {
    if (isEmbeddedInFrame()) return;
    document.addEventListener('fullscreenchange', refreshFullscreenButtonTooltip);
    document.addEventListener('webkitfullscreenchange', refreshFullscreenButtonTooltip);
  }

  /** 独立页：包装主题自带的 toggleFullScreen，防止个别环境下 fullscreenchange 不触发导致气泡不更新 */
  function wrapStandaloneToggleFullScreenForTooltip() {
    if (isEmbeddedInFrame()) return;
    var orig = window.toggleFullScreen;
    if (typeof orig !== 'function' || orig.__dqFullscreenTooltipWrapped) return;
    function wrapped() {
      try {
        return orig.apply(this, arguments);
      } finally {
        requestAnimationFrame(refreshFullscreenButtonTooltip);
        setTimeout(refreshFullscreenButtonTooltip, 0);
        setTimeout(refreshFullscreenButtonTooltip, 350);
      }
    }
    wrapped.__dqFullscreenTooltipWrapped = true;
    window.toggleFullScreen = wrapped;
  }

  function install() {
    bindStandaloneFullscreenTooltipListeners();
    window.addEventListener('load', refreshFullscreenButtonTooltip);

    if (!isEmbeddedInFrame()) {
      wrapStandaloneToggleFullScreenForTooltip();
      refreshFullscreenButtonTooltip();
      return;
    }
    if (typeof window.toggleFullScreen !== 'function') {
      refreshFullscreenButtonTooltip();
      return;
    }
    window.toggleFullScreen = toggleEmbeddedFullscreen;
    syncExpectFromTopDocument();
    attachTopFullscreenListeners();
    refreshFullscreenButtonTooltip();

    document.addEventListener(
      'keydown',
      function (e) {
        if (e.key !== 'Escape') {
          return;
        }
        syncExpectFromTopDocument();
        if (!expectParentFullscreen) {
          return;
        }
        if (trySyncTopExitFromSameOrigin()) {
          expectParentFullscreen = false;
        } else {
          notifyParentExit();
        }
        refreshFullscreenButtonTooltip();
      },
      true
    );
  }

  function scheduleInstall() {
    setTimeout(install, 0);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scheduleInstall);
  } else {
    scheduleInstall();
  }
})();
