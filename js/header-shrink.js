KEEP.initHeaderShrink=()=>{KEEP.utils.headerShrink={headerDom:document.querySelector(".header-wrapper"),isHeaderShrink:!1,init(){this.headerHeight=this.headerDom.getBoundingClientRect().height},headerShrink(){var e=document.body.scrollTop||document.documentElement.scrollTop,r=document.querySelector(".header-wrapper"),{enable:t,header_transparent:a}=KEEP.theme_config.style.first_screen;!this.isHeaderShrink&&e>this.headerHeight?(this.isHeaderShrink=!0,document.body.classList.add("header-shrink"),!0===t&&!0===a&&r.classList.add("transparent-2")):this.isHeaderShrink&&e<=this.headerHeight&&(this.isHeaderShrink=!1,document.body.classList.remove("header-shrink"),!0===t)&&!0===a&&r.classList.remove("transparent-2")},toggleHeaderDrawerShow(){var e=[document.querySelector(".window-mask"),document.querySelector(".menu-bar")];!0===KEEP.theme_config.pjax.enable&&e.push(...document.querySelectorAll(".header-drawer .drawer-menu-list .drawer-menu-item")),e.forEach(e=>{e.addEventListener("click",()=>{document.body.classList.toggle("header-drawer-show")})})}},KEEP.utils.headerShrink.init(),KEEP.utils.headerShrink.headerShrink(),KEEP.utils.headerShrink.toggleHeaderDrawerShow()};