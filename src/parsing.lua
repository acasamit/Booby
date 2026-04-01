local parsing = {}

local MACRO = require("MACRO")

local params

local function help()
	for _, tab in ipairs(params) do
		print(tab[1].."\t\t\t\t"..tab[3])
	end

	os.exit()
end

params = {
	{"-h,--help", function() help() end, "display this message"},
	{"-r,--reload", function() reload_dataset = true end, "reload data.csv and recreate data_train and data_val"},
	{"-l,--layer", function(i) MACRO.HIDED_LAYER = arg[i + 1] end, "specify the number of hided layer", true},
	{"-ls,--layer-size", function(i) MACRO.HIDED_LAYER_SIZE = arg[i + 1] end, "specify the size of hided layers", true},
	{"-e,--epoch", function(i) MACRO.EPOCH = arg[i + 1] end, "specify the number of epoch", true},
	{"-b,--batch", function(i) MACRO.BATCH_SIZE = arg[i + 1] end, "specify the batch size", true},
}

local function split(str)
	local result = {}
	for value in string.gmatch(str, "([^,]+)") do
		table.insert(result, value)
	end
	return result
end

local function find_in_tab(str, tab)
	for _, arg in ipairs(tab) do
		if str == arg then return true end
	end

	return false
end

function parsing.parse_args(args)
	local found = false
	local skip = false

	for i, arg in ipairs(args) do
		if skip then goto continue skip = false end

		for j, param in ipairs(params) do
			local splitted = split(param[1])

			if find_in_tab(arg, splitted) then
				if param[4] then
					assert(args[i + 1], "Error: Missing number")
					skip = true
				end

				param[2](i)
				found = true
			end
		end

		if not found then help() end
		::continue::
		found = false
	end
end

return parsing
