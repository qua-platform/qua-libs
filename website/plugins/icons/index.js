const FaviconsWebpackPlugin = require("favicons-webpack-plugin");
console.log("s")
module.exports = function (context, options) {
  return {
    name: "qua-libs-icon-plugin",
    configureWebpack(config, isServer, utils) {
      return {
        plugins: [new FaviconsWebpackPlugin("src/qua-logo.svg")],
      };
    },
  };
};
