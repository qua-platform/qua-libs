/**
 * Create an index of all examples
 */
const {createTitle} = require("./_text_utils");

const fs = require("fs")
const {collectReadmeFiles} = require("./_collect_files");

const prefix = "../Examples/"
const files = collectReadmeFiles(prefix)

const header = `---
id: examples_index
title: QUA Libs Examples
sidebar_label: Index
slug: /
---
`;

// const sidebar = require("../example_sidebars")
// const lines = []
// Object.entries(sidebar.examples).forEach(([cat, items]) => {
//         lines.push(`## ${cat}`)
//         items.forEach(it => {
//             lines.push(`* [T1](Characterization/T1/)`)
//         })
//     }
// )
const lines = []
let currentGroup = ""
for (let file of files) {
    console.log(file.filePath)
    let parts = file.filePath.split('/')
    if (parts[0] !== currentGroup) {
        currentGroup = parts[0]
        lines.push('')
        lines.push(`### ${createTitle(currentGroup)}`)
        lines.push('')
    }
    lines.push(`* [${createTitle(parts[1])}](${file.filePath})`)
}

fs.writeFileSync(`${prefix}readme.md`, header + lines.join("\n"))
