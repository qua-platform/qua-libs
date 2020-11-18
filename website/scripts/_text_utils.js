module.exports = {
    trimWhitespace(str) {
        return str.replace(/^\s*/, '').replace(/\s*$/, '')
    },
    createSlug(str) {
        return str.replace('_', '-').toLowerCase()
    },
    createTitle(str) {
        return str.replace(/_/g, ' ')
    },
    createId(str) {
        return str.replace(/\//g, '.').toLowerCase()
    },
}
