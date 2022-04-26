/**
 * Create an index of all examples
 */
const fs = require("fs")
const path = require("path")
const {collectReadmeFiles} = require("./_collect_files");
const {loadSidebars} = require("@docusaurus/plugin-content-docs/lib/sidebars");


const prefix = "../examples-old/"
const files = collectReadmeFiles(prefix)

const header = `---
id: examples_index
title: QUA Libs Examples
sidebar_label: Index
slug: /
---
`;

const lines = []
function findFile(id) {
    for(let file of files) {
        if(file.fullId === id) {
            return file
        }
    }
    throw new Error(`Could not find file for id: ${id}`)
}
let sidebars = loadSidebars(path.resolve("sidebars.js"));
const menu = sidebars.examples.slice(1)
for(let l1 of menu) {
    if(l1.type === "doc") {
        const f = findFile(l1.id)
        lines.push(`\n## [${f.sidebar_label}](${f.dirPath}/)`)
    } else if(l1.type === "category") {
        lines.push(`\n## ${l1.label}`)
        for(let l2 of l1.items) {
            if(l2.type === "doc") {
                const f = findFile(l2.id)
                lines.push(`### [${f.sidebar_label}](${f.dirPath}/)`)
            }
        }
    }
}

fs.writeFileSync(`${prefix}readme.md`, header + lines.join("\n"))
