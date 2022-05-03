const math = require("remark-math");
const katex = require("rehype-katex");
const path = require("path");

module.exports = {
  title: "QUA Libraries",
  tagline: "",
  url: "https://qua-platform.github.io",
  baseUrl: "/libs/",
  onBrokenLinks: "throw",
  favicon: "img/qua-logo.ico",
  organizationName: "qua-platform",
  projectName: "qua-libs",
  themeConfig: {
    navbar: {
      title: "QUA Libraries",
      logo: {
        alt: "QUA",
        src: "img/qua-logo.svg",
      },
      items: [
        {
          to: "examples/",
          activeBasePath: "examples",
          label: "Examples",
          position: "left",
        },
        {
          to: "contributing/",
          activeBasePath: "contributing",
          label: "Contributing",
          position: "left",
        },
        {
          href: "https://github.com/qua-platform/qua-libs",
          label: "GitHub",
          position: "right",
        },
        {
          href: "https://discord.gg/7FfhhpswbP",
          label: "Discord",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [],
      copyright: `Copyright © ${new Date().getFullYear()} QM, Inc. Built with Docusaurus.`,
    },
  },
  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X",
      crossorigin: "anonymous",
    },
  ],
  plugins: [path.resolve(__dirname, "plugins/icons")],
  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          routeBasePath: "examples",
          path: "../examples-old",
          sidebarPath: require.resolve("./sidebars.js"),
          // Please change this to your repo.
          editUrl:
            "https://github.com/qua-platform/qua-libs/edit/main/website/",
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
};
