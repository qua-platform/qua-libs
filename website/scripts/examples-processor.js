/**
 * Rewrite the examples readme.md files to have the id and slug matching the directory structure
 */
const {trimWhitespace} = require("./_text_utils");

const fs = require("fs")
const {collectReadmeFiles} = require("./_collect_files");

const prefix = "../Examples/"
const files = collectReadmeFiles(prefix)

const sepRegex = /^\s*---\s*$/

for (let file of files) {
    let filePath = `${prefix}${file.filePath}`;
    const content = fs.readFileSync(filePath).toString()
    const lines = content.split("\n")
    let i = 0;
    let firstSepEncountered = false
    const originalProps = {}
    while (i < lines.length) {
        if (!firstSepEncountered) {
            if (!sepRegex.test(lines[i])) {
                break
            }
            firstSepEncountered = true
        } else {
            if (sepRegex.test(lines[i])) {
                i++
                break
            } else {
                let parts = lines[i].split(":");
                originalProps[parts[0]] = trimWhitespace(parts[1])
            }
        }
        i++
    }
    const props = {
        ...originalProps,
        id: file.id,
        title: file.title,
        sidebar_label: file.title,
        slug: file.slug
    }
    const header = Object.entries(props).map(([k, v]) => `${k}: ${v}`)

    const newLines = ["---", ...header, "---", ...lines.slice(i)]

    fs.writeFileSync(filePath, newLines.join("\n"))
}
