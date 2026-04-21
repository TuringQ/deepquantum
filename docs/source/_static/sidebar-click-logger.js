/*
 * @Author: ZhengPing zhengping@turingq.com
 * @Date: 2026-03-16 16:09:11
 * @Description: 左侧菜单栏与 iframe 父窗口通信
 *
 * 当文档在 iframe 中打开时：一级菜单点击失效，通过 postMessage 向父窗口传递导航信息
 * 独立打开时：保持默认原生跳转，并 POST 当前 path 到 case-visit-detail 接口
 * Copyright (c) 2026 zhengping@turingq.com. All Rights Reserved.
 */

(function () {

  const POST_MESSAGE_ORIGIN = '*';
  const MESSAGE_TYPE_NAVIGATE = 'DEEPQUANTUM_NAVIGATE';
  const MESSAGE_TYPE_REPORT_SUCCESS = 'DEEPQUANTUM_CASE_VISIT_REPORTED';
  /** 父页面「回到顶部」按钮 → iframe 内文档同步置顶（与 parentPage.vue 约定一致） */
  const MESSAGE_TYPE_SCROLL_TO_TOP = 'DEEPQUANTUM_SCROLL_TO_TOP';
  /** iframe 内主内容滚动深度 → 父页 BackToTop 是否与「子页滚动 > 阈值」联动显示 */
  const MESSAGE_TYPE_IFRAME_SCROLL_DEPTH = 'DEEPQUANTUM_IFRAME_SCROLL_DEPTH';
  /** 与 backToTop.vue 中 SCROLL_THRESHOLD 保持一致（一键置顶） */
  var IFRAME_SCROLL_DEPTH_THRESHOLD = 250;
  /** 父页折叠 Banner（与 parentPage 约定 200px） */
  var IFRAME_BANNER_COLLAPSE_THRESHOLD = 200;
  const CASE_VISIT_API = 'https://dq-back-api-test.turingq.com/api/deepquantum/case-visit-detail/add';

  const SELECTORS = {
    SIDEBAR: '.bd-links',
    MENU_ITEM: 'li.toctree-l1',
    MENU_LINK: 'a.reference.internal',
    PARENT_SECTION: 'li.toctree-l0',
    CAPTION: '.caption-text',
    /** PyData navbar-logo.html：主题头部 Logo，href 常为 # 时原先不会走侧栏 NAVIGATE */
    NAVBAR_LOGO: 'a.navbar-brand.logo',
    /** 页脚 prev/next（sphinx_book_theme / pydata 常见 class） */
    PREV_NEXT: 'a.left-prev, a.right-next'
  };

  /**
   * 检查当前页面是否被 iframe 嵌入
   */
  function isInIframe() {
    return window.self !== window.top;
  }

  /**
   * 点击左侧目录后把主内容滚到顶（window + PyData 常见滚动容器）
   */
  function scrollDocsContentToTop() {
    try {
      if ('scrollRestoration' in history) {
        history.scrollRestoration = 'manual';
      }
    } catch (e) {
      /* ignore */
    }
    try {
      window.scrollTo(0, 0);
      var se = document.scrollingElement;
      if (se) se.scrollTop = 0;
      if (document.documentElement) document.documentElement.scrollTop = 0;
      if (document.body) document.body.scrollTop = 0;
      var selectors =
        'main.bd-main, main#main-content, main, .bd-main, .bd-content, .bd-page-width, article.bd-article';
      document.querySelectorAll(selectors).forEach(function (el) {
        el.scrollTop = 0;
      });
    } catch (e) {
      /* ignore */
    }
  }

  /**
   * 上报 Case 访问：POST 到 case-visit-detail 接口
   * @param {string} path - 要上报的访问路径
   * @returns {Promise<string>} 成功时 resolve 上报的 path，失败时 reject
   */
  function reportCaseVisit(path) {
    const url = path || window.location.href;
    const payload = JSON.stringify({ path: url });
    return fetch(CASE_VISIT_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
      .then(function (res) {
        if (!res.ok) throw new Error('reportCaseVisit failed: ' + res.status);
        return url;
      })
      .catch(function (err) {
        console.warn('[DEEPQUANTUM] reportCaseVisit failed:', err);
        throw err;
      });
  }

  /**
   * 向父窗口发送 postMessage
   * @param {Object} payload - 消息内容
   */
  function postToParent(payload) {
    if (!isInIframe()) return;
    try {
      window.parent.postMessage(payload, POST_MESSAGE_ORIGIN);
    } catch (err) {
      console.warn('[DEEPQUANTUM] postMessage failed:', err);
    }
  }

  function isDocSidebarEl(el) {
    if (!el || !el.classList) return false;
    return (
      el.classList.contains('bd-sidebar-primary') ||
      el.classList.contains('bd-sidebar-secondary') ||
      el.id === 'pst-primary-sidebar' ||
      el.id === 'pst-secondary-sidebar'
    );
  }

  /**
   * 主文章列垂直滚动量（与 PyData 内层滚动容器一致，避免仅用 window.scrollY）
   */
  function getMainContentScrollY() {
    var vals = [0];
    var wy = window.scrollY || window.pageYOffset;
    if (wy) vals.push(wy);
    var se = document.scrollingElement;
    if (se) vals.push(se.scrollTop || 0);
    if (document.documentElement) vals.push(document.documentElement.scrollTop || 0);
    if (document.body) vals.push(document.body.scrollTop || 0);
    var root = document.querySelector('main#main-content') || document.querySelector('main.bd-main');
    if (root) {
      var el = root;
      while (el && el !== document.body) {
        if (!isDocSidebarEl(el)) {
          vals.push(el.scrollTop || 0);
        }
        el = el.parentElement;
      }
    }
    return Math.max.apply(null, vals);
  }

  function collectMainColumnScrollTargets() {
    var targets = [];
    var seen = new Set();
    function add(t) {
      if (!t || seen.has(t)) return;
      seen.add(t);
      targets.push(t);
    }
    add(window);
    var root = document.querySelector('main#main-content') || document.querySelector('main.bd-main');
    if (root) {
      var el = root;
      while (el && el !== document.body) {
        if (!isDocSidebarEl(el)) {
          try {
            if (el.scrollHeight > el.clientHeight + 1) {
              var oy = window.getComputedStyle(el).overflowY;
              if (oy === 'auto' || oy === 'scroll' || oy === 'overlay') {
                add(el);
              }
            }
          } catch (e) {
            /* ignore */
          }
        }
        el = el.parentElement;
      }
    }
    return targets;
  }

  var iframeScrollDepthBindings = [];

  function tearDownIframeScrollDepthListeners() {
    iframeScrollDepthBindings.forEach(function (rec) {
      try {
        rec.target.removeEventListener('scroll', rec.sync);
      } catch (e) {
        /* ignore */
      }
    });
    iframeScrollDepthBindings = [];
  }

  /**
   * iframe 内滚动深度 → 父页：一键置顶(250px) / 折叠 Banner(200px)
   * 仅在 pastThreshold 或 bannerCollapsed 变化时发送，减少消息量
   */
  function initIframeScrollDepthReporting() {
    if (!isInIframe()) return;

    var lastPast = null;
    var lastBannerCollapsed = null;

    function syncIframeScrollDepth() {
      var y = getMainContentScrollY();
      var past = y > IFRAME_SCROLL_DEPTH_THRESHOLD;
      var bannerCollapsed = y > IFRAME_BANNER_COLLAPSE_THRESHOLD;
      if (lastPast === past && lastBannerCollapsed === bannerCollapsed) return;
      lastPast = past;
      lastBannerCollapsed = bannerCollapsed;
      postToParent({
        type: MESSAGE_TYPE_IFRAME_SCROLL_DEPTH,
        scrollY: Math.round(y),
        pastThreshold: past,
        bannerCollapsed: bannerCollapsed
      });
    }

    function bindScrollRoots() {
      tearDownIframeScrollDepthListeners();
      collectMainColumnScrollTargets().forEach(function (target) {
        target.addEventListener('scroll', syncIframeScrollDepth, { passive: true });
        iframeScrollDepthBindings.push({ target: target, sync: syncIframeScrollDepth });
      });
      syncIframeScrollDepth();
    }

    bindScrollRoots();

    var resizeTimer = null;
    window.addEventListener('resize', function () {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(bindScrollRoots, 150);
    });

    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () {
        bindScrollRoots();
      }).catch(function () {});
    }
  }

  /**
   * 判断实际点击的 link 是否属于 Demos 父级章节（一级 toctree-l0）
   * @param {HTMLAnchorElement} link - 实际被点击的菜单链接
   * @returns {boolean}
   */
  function isLinkUnderDemos(link) {
    const parentSectionEl = link.closest(SELECTORS.PARENT_SECTION);
    if (!parentSectionEl) return false;
    const captionEl = parentSectionEl.querySelector(SELECTORS.CAPTION);
    return (captionEl && captionEl.textContent.trim()) === 'Demos';
  }

  /**
   * 与 sidebar-viewport-sync.js 一致：整页处于 Demos/Cases 文档上下文中（用于 prev/next 等同侧栏的 preventDefault 策略）
   */
  function isDemosOrCasesDocsContext() {
    try {
      if (document.body && document.body.classList.contains('demos-page')) return true;
    } catch (e) {
      /* ignore */
    }
    var path = (window.location.pathname || '').replace(/\/$/, '');
    return path.indexOf('/demos/') >= 0 || path.indexOf('/cases/') >= 0;
  }

  /**
   * 从实际点击的链接提取上下文信息（支持一级、二级、三级菜单）
   * @param {HTMLAnchorElement} link - 实际被点击的菜单链接
   * @returns {Object} 菜单上下文
   */
  function getMenuContext(link) {
    const parentSectionEl = link.closest(SELECTORS.PARENT_SECTION);
    const captionEl = parentSectionEl && parentSectionEl.querySelector(SELECTORS.CAPTION);
    const parentSectionName = (captionEl && captionEl.textContent.trim()) || '';

    return {
      url: link.href,
      linkText: link.textContent.trim(),
      parentSection: parentSectionName,
      currentUrl: window.location.href,
      currentTitle: document.title
    };
  }

  /**
   * 处理菜单点击（一级、二级、三级及任意层级）
   * 使用实际点击的 link 确保传递正确的路由
   * @param {MouseEvent} event
   */
  function handleMenuClick(event) {
    try {
      const link = event.target.closest(SELECTORS.MENU_LINK);
      if (!link || !link.closest(SELECTORS.SIDEBAR)) return;

      scrollDocsContentToTop();

      if (isLinkUnderDemos(link)) {
        event.preventDefault();
        const targetUrl = link.href;
        const context = getMenuContext(link);
        if (isInIframe()) {
          postToParent({ type: MESSAGE_TYPE_NAVIGATE, ...context });
        } else {
          window.location.href = targetUrl;
        }
      } else if (isInIframe()) {
        postToParent({
          type: MESSAGE_TYPE_NAVIGATE,
          ...getMenuContext(link)
        });
      }
    } catch (err) {
      console.warn('[DEEPQUANTUM] handleMenuClick failed:', err);
    }
  }

  /**
   * 修复 Demos/Cases 子页面侧边栏 caption 显示为小写的问题
   */
  function fixCasesCaption() {
    try {
      document.querySelectorAll(SELECTORS.CAPTION).forEach((el) => {
        const t = el.textContent.trim();
        if (t === 'cases') el.textContent = 'Cases';
        else if (t === 'demos') el.textContent = 'Demos';
      });
    } catch (err) {
      console.warn('[DEEPQUANTUM] fixCasesCaption failed:', err);
    }
  }

  /**
   * 初始化侧边栏点击监听
   */
  function initSidebarClickLogger() {
    const sidebar = document.querySelector(SELECTORS.SIDEBAR);
    if (!sidebar) return;

    sidebar.addEventListener('click', handleMenuClick);
  }

  /**
   * 顶部主题 Logo（navbar-brand logo）：在 iframe 内向父页补发 DEEPQUANTUM_NAVIGATE（与侧栏一致）
   * href="#" / 空 时仅滚文档顶并上报当前页 URL（无 hash），便于父页与同 URL 分支联动。
   */
  function handleNavbarBrandLogoClick(event) {
    if (!isInIframe()) return;
    var link = event.target && event.target.closest && event.target.closest(SELECTORS.NAVBAR_LOGO);
    if (!link || link.tagName !== 'A') return;

    var urlNoHash = '';
    try {
      urlNoHash = String(link.href || '').split('#')[0];
    } catch (e) {
      return;
    }
    if (!urlNoHash) {
      try {
        urlNoHash = String(window.location.href || '').split('#')[0];
      } catch (e2) {
        return;
      }
    }

    var rawHref = (link.getAttribute('href') || '').trim();
    if (rawHref === '#' || rawHref === '') {
      scrollDocsContentToTop();
    }

    postToParent({
      type: MESSAGE_TYPE_NAVIGATE,
      url: urlNoHash,
      linkText: (link.textContent || '').trim() || 'Logo',
      parentSection: '',
      currentUrl: window.location.href,
      currentTitle: document.title
    });
  }

  function initNavbarBrandLogoBridge() {
    document.addEventListener('click', handleNavbarBrandLogoClick, true);
  }

  /**
   * 页脚「上一页 / 下一页」：与侧栏一致向父页 DEEPQUANTUM_NAVIGATE；Demos/Cases 下 iframe 内阻止默认由父页改 iframe src
   */
  function handlePrevNextClick(event) {
    try {
      var link = event.target && event.target.closest && event.target.closest(SELECTORS.PREV_NEXT);
      if (!link || link.tagName !== 'A') return;

      scrollDocsContentToTop();

      var context = {
        url: link.href,
        linkText: (link.textContent || '').trim(),
        parentSection: '',
        currentUrl: window.location.href,
        currentTitle: document.title
      };

      if (isDemosOrCasesDocsContext()) {
        event.preventDefault();
        if (isInIframe()) {
          postToParent({ type: MESSAGE_TYPE_NAVIGATE, ...context });
        } else {
          window.location.href = link.href;
        }
        return;
      }

      if (isInIframe()) {
        postToParent({ type: MESSAGE_TYPE_NAVIGATE, ...context });
      }
    } catch (err) {
      console.warn('[DEEPQUANTUM] handlePrevNextClick failed:', err);
    }
  }

  function initPrevNextNavBridge() {
    document.addEventListener('click', handlePrevNextClick, true);
  }

  /**
   * 父窗口 postMessage 要求文档滚到顶部（见 _static/parentPage.vue）
   */
  function initParentScrollToTopMessage() {
    window.addEventListener('message', function (event) {
      try {
        if (!isInIframe()) return;
        if (event.source !== window.parent) return;
        var d = event.data;
        if (!d || typeof d !== 'object' || d.type !== MESSAGE_TYPE_SCROLL_TO_TOP) return;
        scrollDocsContentToTop();
      } catch (err) {
        console.warn('[DEEPQUANTUM] parent scroll-to-top message failed:', err);
      }
    });
  }

  function init() {
    try {
      fixCasesCaption();
      initSidebarClickLogger();
      initNavbarBrandLogoBridge();
      initPrevNextNavBridge();
      initParentScrollToTopMessage();
      initIframeScrollDepthReporting();
      reportCaseVisit(window.location.href)
        .then(function (reportedPath) {
          if (isInIframe()) {
            postToParent({
              type: MESSAGE_TYPE_REPORT_SUCCESS,
              path: reportedPath,
              url: window.location.href,
              currentTitle: document.title
            });
          }
        })
        .catch(function () {
          /* 已在 reportCaseVisit 中 log */
        });
    } catch (err) {
      console.warn('[DEEPQUANTUM] init failed:', err);
    }
  }

  try {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  } catch (err) {
    console.warn('[DEEPQUANTUM] script load failed:', err);
  }
})();
