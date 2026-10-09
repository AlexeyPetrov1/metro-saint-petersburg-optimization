-- Неразрывный пробел между числом и единицей («5,51 %», «0,80 п. п.», «2 ч»), чтобы перенос строки их не разрывал.
local units = {["%"] = true, ["п."] = true, ["ч"] = true, ["ч,"] = true, ["ч."] = true, ["тыс."] = true,
               ["мин"] = true, ["суток"] = true, ["сут."] = true, ["мм"] = true}
local function unit_start(s)
  if units[s] then return true end
  return s:sub(1, 1) == "%"
end
function Inlines(inl)
  for i = 2, #inl - 1 do
    local a, b, c = inl[i - 1], inl[i], inl[i + 1]
    if b.t == "Space" and a.t == "Str" and c.t == "Str" and a.text:match("[%d%)%]]$") and unit_start(c.text) then
      inl[i] = pandoc.Str("\u{00A0}")
    end
  end
  return inl
end
