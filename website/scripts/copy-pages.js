const fs = require("fs")
const path = require("path")

const sourcePath = path.resolve(__dirname, "../../CONTRIBUTING.md")
const targetPath = path.resolve(__dirname, "../src/pages/contributing.md")
let content = fs.readFileSync(sourcePath).toString()
let modified = content.replace(/\[(.+?)\]\((.+?)\)/g, `[$1](https://github.com/qua-platform/qua-libs/blob/main/$2)`)
fs.writeFileSync(targetPath, modified)