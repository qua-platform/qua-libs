const FaviconsWebpackPlugin = require("favicons-webpack-plugin");
const path = require("path")

module.exports = function (context, options) {
  return {
    name: "qua-libs-icon-plugin",
    configureWebpack(config, isServer, utils) {
      return {
        plugins: [new FaviconsWebpackPlugin(path.resolve("src/qua-logo.svg"))],
      };
    },
  };
};
