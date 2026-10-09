-- PDF: инлайн-код — \texttt с точками переноса после «/», «.», «_», «:», чтобы длинные пути и имена переносились.
-- Работает и в подписях (в отличие от \path).
local esc = {["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$", ["&"] = "\\&",
             ["#"] = "\\#", ["^"] = "\\textasciicircum{}", ["_"] = "\\_", ["%"] = "\\%", ["~"] = "\\textasciitilde{}"}
if FORMAT:match("latex") then
  function Code(el)
    local out = {}
    for _, cp in utf8.codes(el.text) do
      local ch = utf8.char(cp)
      local t = esc[ch] or ch
      if ch == "/" or ch == "." or ch == "_" or ch == ":" then t = t .. "\\allowbreak{}" end
      table.insert(out, t)
    end
    return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
  end
end
