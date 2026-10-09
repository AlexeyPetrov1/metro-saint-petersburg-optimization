-- PDF: аккуратная вёрстка таблиц и подзаголовков.
-- 1. Компактные таблицы (до 22 строк и 1 300 знаков) — обычный tabular в неплавающей table [H], а не longtable: longtable не учитывает
--    плавающие рисунки на той же странице и выталкивает текст за нижнее поле. Большие таблицы остаются longtable.
-- 2. Абзац-подзаголовок («**Почему ….**») не остаётся один внизу страницы.
if not FORMAT:match("latex") then return {} end

local MAX_ROWS = 22

local function needspace(lines)
  return pandoc.RawBlock("latex", string.format("\\needspace{%d\\baselineskip}", lines))
end

local function lead_in(para)
  local c = para.content
  return #c >= 1 and c[1].t == "Strong" and pandoc.utils.stringify(para):len() < 160
end

local function rows(tbl)
  local n = #tbl.head.rows
  for _, body in ipairs(tbl.bodies) do n = n + #body.body end
  return n
end

-- longtable от pandoc → table[H] + tabular; если формат неожиданный — nil (останется longtable)
local function as_tabular(tbl)
  local tex = pandoc.write(pandoc.Pandoc({tbl}), "latex")
  local spec = tex:match("\\begin{longtable}%[%](%b{})")
  local caption = tex:match("(\\caption.-)\\tabularnewline")
  local head = tex:match("\\tabularnewline\n(.-)\\endfirsthead")
  local body = tex:match("\\endlastfoot\n(.-)\\end{longtable}")
  if not (spec and caption and head and body) then return nil end
  -- широкие таблицы (от 10 колонок) — мельче и с узкими отступами, чтобы помещались на книжной странице
  local size = #tbl.colspecs >= 10 and "\\scriptsize\\setlength{\\tabcolsep}{3pt}" or
               "\\footnotesize\\setlength{\\tabcolsep}{4pt}"
  return pandoc.RawBlock("latex", table.concat({
    "\\begin{table}[H]",
    "\\centering" .. size .. "\\renewcommand{\\arraystretch}{1.15}",
    caption,
    "\\begin{tabular}" .. spec,
    head .. body .. "\\bottomrule\\noalign{}",
    "\\end{tabular}",
    "\\end{table}"}, "\n"))
end

-- высокая текстовая таблица (много текста в ячейках) — пусть остаётся longtable и делится по страницам
local MAX_CHARS = 1300
local function compact(tbl)
  local wide = #tbl.colspecs >= 10      -- широкие числовые таблицы — всегда одним блоком мелким шрифтом
  return rows(tbl) <= MAX_ROWS and (wide or pandoc.utils.stringify(tbl):len() <= MAX_CHARS)
end

function Blocks(blocks)
  local out = pandoc.List()
  for _, b in ipairs(blocks) do
    if b.t == "Table" and compact(b) then
      out:insert(as_tabular(b) or b)
    elseif b.t == "Table" then
      out:insert(pandoc.RawBlock("latex", "\\FloatBarrier"))
      out:insert(b)
    elseif b.t == "Para" and lead_in(b) and not (#out > 0 and out[#out].t == "Header") then
      -- сразу после заголовка запас не нужен: его уже даёт заголовок, а лишний \needspace оторвал бы абзац от него
      out:insert(needspace(6))
      out:insert(b)
    else
      out:insert(b)
    end
  end
  return out
end
