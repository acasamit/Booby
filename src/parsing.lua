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
	{"-r,--reload", function() MACRO.RELOAD_DATASET = true end, "reload data.csv and recreate data_train and data_val"},
	{"-t,--train", function() MACRO.TRAIN = true end, "train the model event if there is a best model file"},
	{"-l,--layer", function(value) MACRO.HIDED_LAYER = value end, "specify the number of hided layer", true},
	{"-ls,--layer-size", function(value) MACRO.HIDED_LAYER_SIZE = value end, "specify the size of hided layers", true},
	{"-e,--epoch", function(value) MACRO.EPOCH = value end, "specify the number of epoch", true},
	{"-b,--batch", function(value) MACRO.BATCH_SIZE = value end, "specify the batch size", true},
	{"-o,--output", function(value) MACRO.MODEL_NAME = value end, "specify the output model name", true},
	{"-m,--model", function(value) MACRO.MODEL_USED = value end, "choose the used model", true},
	{"-p,--predict", function(value) MACRO.PREDICT = value end, "choose a data file to predict", true},
	{"-lr,--learning-rate", function(value) MACRO.LEARNING_RATE = value end, "choose a data file to predict", true},
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
		if skip then skip = false goto continue end

		for j, param in ipairs(params) do
			local splitted = split(param[1])

			if find_in_tab(arg, splitted) then
				if param[4] then
					local value = args[i + 1]
					assert(value, "Error: Missing value for " .. arg)
					param[2](value)
					skip = true
				else
					param[2]()
				end

				found = true
			end
		end

		if not found then help() end
		::continue::
		found = false
	end
end

return parsing
