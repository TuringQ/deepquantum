/*
 * @Author: ZhengPing zhengping@turingq.com
 * @Date: 2026-03-19 17:01:37
 * @LastEditors: pandorping 1159927992@qq.com
 * @FilePath: /deepquantum/docs/source/_static/sidebar-viewport-sync.js
 * @Description:
 *
 * Copyright (c) 2026 by zhengping@turingq.com, All Rights Reserved.
 */
/**
 * 根据视口宽度同步左侧栏 checkbox 状态，解决 sphinx_book_theme 与 pydata 主题 checkbox 语义相反的问题。
 *
 * 语义冲突：
 * - 大屏 Web (≥992px, sphinx_book_theme)：checked=侧栏收起，unchecked=侧栏展开
 * - 小屏设备 (<992px, pydata)：checked=抽屉打开+overlay，unchecked=抽屉关闭
 *
 * 完整交互规则：
 * 1. 大屏 Web：Demos 默认收起，其他子页面默认展开
 * 2. 小屏设备（手机/平板）：所有子页面默认收起，无差异化，仅可通过汉堡按钮手动展开
 *
 * 滚动与侧栏：
 * - 浏览器在纵向滚动时常因地址栏显隐等触发 resize，但 innerWidth 不变；若仍调用 sync，
 *   会把用户已手动展开的 Demos 侧栏再次强制收起。故仅在 innerWidth 变化时做宽度相关同步。
 * - 大屏 Demos/Cases 下用户手动切换侧栏后，不再被「默认收起」覆盖，直至跨越992 断点（横竖屏等）。
 */
(function () {
  var SIDEBAR_BREAKPOINT = 992;

  var lastWidthForSidebar = typeof window !== 'undefined' ? window.innerWidth : 0;
  /** 大屏 Demos/Cases 下用户是否手动改过侧栏（避免被默认策略反复盖掉） */
  var userDemosSidebarManual = false;

  /**
   * 判断当前页是否为 demos 示例页面（pathname 包含 /demos/；保留 /cases/ 以兼容旧书签）
   */
  function isDemosOrCasesPage() {
    if (document.body && document.body.classList.contains('demos-page')) return true;
    var path = (window.location.pathname || '').replace(/\/$/, '');
    return path.indexOf('/demos/') >= 0 || path.indexOf('/cases/') >= 0;
  }

  function isLargeScreen() {
    return window.innerWidth >= SIDEBAR_BREAKPOINT;
  }

  var SIDEBAR_HIDDEN_CLASS = 'pst-sidebar-hidden';
  var PRIMARY_SIDEBAR_SEL = '#pst-primary-sidebar';

  function wirePrimarySidebarCheckbox() {
    var cb = document.getElementById('pst-primary-sidebar-checkbox');
    if (!cb || cb.dataset.dqSidebarViewportSyncWired) return;
    cb.dataset.dqSidebarViewportSyncWired = '1';
    cb.addEventListener('change', function () {
      if (isDemosOrCasesPage() && isLargeScreen()) {
        userDemosSidebarManual = true;
      }
    });
  }

  function syncSidebarState() {
    var isDemosCases = isDemosOrCasesPage();
    var largeScreen = isLargeScreen();
    // 大屏且 demos 示例页：侧栏默认收起；其他：展开
    var shouldBeHidden = largeScreen && isDemosCases;

    var checkbox = document.getElementById('pst-primary-sidebar-checkbox');
    if (checkbox) {
      if (userDemosSidebarManual && isDemosCases && largeScreen) {
        return;
      }
      if (checkbox.checked !== shouldBeHidden) {
        checkbox.checked = shouldBeHidden;
      }
      return;
    }

    // 无 checkbox 时直接操作侧栏 class（sphinx-book-theme 等主题）
    var sidebar = document.querySelector(PRIMARY_SIDEBAR_SEL);
    if (!sidebar) return;
    if (userDemosSidebarManual && isDemosCases && largeScreen) {
      return;
    }
    var hasHidden = sidebar.classList.contains(SIDEBAR_HIDDEN_CLASS);
    if (shouldBeHidden && !hasHidden) {
      sidebar.classList.add(SIDEBAR_HIDDEN_CLASS);
    } else if (!shouldBeHidden && hasHidden) {
      sidebar.classList.remove(SIDEBAR_HIDDEN_CLASS);
    }
  }

  function onWindowResize() {
    var w = window.innerWidth;
    if (w === lastWidthForSidebar) {
      return;
    }
    var prevLarge = lastWidthForSidebar >= SIDEBAR_BREAKPOINT;
    var nowLarge = w >= SIDEBAR_BREAKPOINT;
    lastWidthForSidebar = w;
    if (prevLarge !== nowLarge) {
      userDemosSidebarManual = false;
    }
    syncSidebarState();
  }

  function init() {
    lastWidthForSidebar = window.innerWidth;
    userDemosSidebarManual = false;
    wirePrimarySidebarCheckbox();
    syncSidebarState();
    setTimeout(function () {
      wirePrimarySidebarCheckbox();
      syncSidebarState();
    }, 20);
    window.addEventListener('resize', onWindowResize);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
