/**
 * 产品站头部/页脚交互：移动端菜单、搜索层、滚动样式（静态文档无 Vue）
 * 独立打开：仅头部等产品壳层；iframe 嵌入（html.dq-chrome-embed）：仅页脚（见 site-chrome.css），并绑定页脚滚动样式。
 */
(function () {
  /**
   * GitHub 仓库：pointerdown 捕获阶段早于 mousedown/click 与默认导航；此处同步 reset全部 Tooltip，
   * 先从 DOM 移除浮层再 dispose/重建，气泡在跳转前即消失。键盘 Enter/Space 同理。
   * 切走标签（hidden）时同步摘掉浮层并 dispose，避免后台页仍挂着 .tooltip；切回（visible）同步 reset，不用 setTimeout。
   */
  var TOOLTIP_DELAY = { show: 500, hide: 100 };

  function removeTooltipDomEverywhere() {
    document.querySelectorAll('.tooltip').forEach(function (tip) {
      tip.remove();
    });
  }

  /** 仅清 DOM + dispose，不重建（文档标签切入后台时调用） */
  function disposeAllTooltipsOnly() {
    removeTooltipDomEverywhere();
    if (typeof bootstrap === 'undefined' || !bootstrap.Tooltip) return;
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
      var inst = bootstrap.Tooltip.getInstance(el);
      if (inst) {
        try {
          inst.dispose();
        } catch (err) {}
      }
      el.removeAttribute('aria-describedby');
    });
  }

  /** 清掉残留浮层并重建实例（切回标签 / bfcache / 点 GitHub 后当前页仍在前台时） */
  function resetAllBootstrapTooltips() {
    removeTooltipDomEverywhere();
    if (typeof bootstrap === 'undefined' || !bootstrap.Tooltip) return;
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
      var inst = bootstrap.Tooltip.getInstance(el);
      if (inst) {
        try {
          inst.dispose();
        } catch (err) {}
      }
      el.removeAttribute('aria-describedby');
    });
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
      try {
        new bootstrap.Tooltip(el, { delay: TOOLTIP_DELAY });
      } catch (err) {}
    });
  }

  function hideAllTooltipsBeforeRepoNavigate(e) {
    var t = e.target;
    var a = t && t.closest ? t.closest('a.btn-source-repository-button') : null;
    if (!a) return;
    resetAllBootstrapTooltips();
  }

  document.addEventListener('pointerdown', hideAllTooltipsBeforeRepoNavigate, true);

  document.addEventListener(
    'keydown',
    function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      var el = document.activeElement;
      if (!el || !el.classList || !el.classList.contains('btn-source-repository-button')) return;
      resetAllBootstrapTooltips();
    },
    true
  );

  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'hidden') {
      disposeAllTooltipsOnly();
      return;
    }
    if (document.visibilityState === 'visible') {
      resetAllBootstrapTooltips();
    }
  });

  window.addEventListener('pageshow', function () {
    resetAllBootstrapTooltips();
  });

  /**
   * 嵌在父壳 iframe 中时，子文档自身的 visibility 与顶层标签切换可能不同步；
   * 由父页 postMessage 同步顶层 document.visibilityState（见 DeepQuantumCaseDetail/index.vue）。
   */
  var MSG_PARENT_DOC_VISIBILITY = 'DEEPQUANTUM_PARENT_DOCUMENT_VISIBILITY';

  window.addEventListener('message', function (e) {
    var data = e.data;
    if (!data || data.type !== MSG_PARENT_DOC_VISIBILITY) return;
    try {
      if (window.parent === window.self) return;
      if (e.source !== window.parent) return;
    } catch (err) {
      return;
    }
    if (data.state === 'hidden') {
      disposeAllTooltipsOnly();
    } else if (data.state === 'visible') {
      resetAllBootstrapTooltips();
    }
  });

  function qs(sel, root) {
    return (root || document).querySelector(sel);
  }

  function isEmbeddedInFrame() {
    try {
      return window.self !== window.top;
    } catch (e) {
      return true;
    }
  }

  function onScroll() {
    var y = window.scrollY || 0;
    var header = qs('#dq-site-header');
    var footer = qs('#dq-site-footer');
    if (header) header.classList.toggle('scrolled', y > 50);
    if (footer) footer.classList.toggle('scrolled', y > 100);
  }

  function openModal(el) {
    if (!el) return;
    el.classList.add('dq-is-open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal(el) {
    if (!el) return;
    el.classList.remove('dq-is-open');
    document.body.style.overflow = '';
  }

  function bind() {
    if (isEmbeddedInFrame()) {
      document.documentElement.classList.add('dq-chrome-embed');
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
      return;
    }

    var searchModal = qs('#dq-search-modal');
    var mobilePanel = qs('#dq-mobile-panel');

    document.querySelectorAll('.dq-search-open').forEach(function (btn) {
      btn.addEventListener('click', function () {
        openModal(searchModal);
        setTimeout(function () {
          var input = searchModal && searchModal.querySelector('.modal-field');
          if (input) input.focus();
        }, 100);
      });
    });

    document.querySelectorAll('.dq-search-close').forEach(function (btn) {
      btn.addEventListener('click', function () {
        closeModal(searchModal);
      });
    });

    if (searchModal) {
      searchModal.addEventListener('click', function (e) {
        if (e.target === searchModal) closeModal(searchModal);
      });
    }

    var openMb = qs('.dq-mobile-menu-open');
    var closeMb = qs('.dq-mobile-menu-close');
    if (openMb && mobilePanel) {
      openMb.addEventListener('click', function () {
        openModal(mobilePanel);
      });
    }
    if (closeMb && mobilePanel) {
      closeMb.addEventListener('click', function () {
        closeModal(mobilePanel);
      });
    }
    if (mobilePanel) {
      mobilePanel.addEventListener('click', function (e) {
        if (e.target.tagName === 'A') closeModal(mobilePanel);
      });
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
