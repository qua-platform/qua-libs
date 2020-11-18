const { createSlug, createTitle, createId } = require("./_text_utils");
const glob = require("glob");
const fs = require("fs");
const { trimWhitespace } = require("./_text_utils");

const keyValMatcher = /^(\w+):\s*(.+?)\s*$/;
function getMeta(content) {
  let lines = content.split("\n");
  let foundFirstSep = false;
  let props = {};
  for (let line of lines) {
    line = trimWhitespace(line);
    if (!foundFirstSep) {
      if (line === "---") {
        foundFirstSep = true;
      } else {
        return {}
      }
    } else {
      if (line === "---") {
        break;
      }
      let m = keyValMatcher.exec(line);
      if (m) {
        props[m[1]] = m[2];
      }
    }
  }
  return props
}

module.exports.collectReadmeFiles = (prefix, ignoreRoot = true) =>
  glob
    .sync(`${prefix}**/readme.md`)
    // strip prefix
    .map((it) => it.substr(prefix.length))
    .filter((it) => !ignoreRoot || it !== "readme.md")
    .map((it) => {
      let fileContent = fs.readFileSync(prefix + it).toString();
      let meta = getMeta(fileContent);

      const noFile = it.replace(/\/readme\.md$/, "");
      let pathParts = noFile.split("/");
      let lastPart = pathParts[pathParts.length - 1];
      let beforeLastPart = pathParts[pathParts.length - 2];
      let id = meta.id || createId(lastPart);
      let fullId = pathParts.join("/") + "/" + id;
      let slug = meta.slug || createSlug(lastPart);
      let title = meta.title || createTitle(lastPart);
      let sidebar_label = meta.sidebar_label || createTitle(lastPart);
      let group = beforeLastPart ? createTitle(beforeLastPart) : "";
      return {
        filePath: it,
        dirPath: noFile,
        id,
        fullId,
        title,
        sidebar_label,
        group,
        slug,
      };
    });
