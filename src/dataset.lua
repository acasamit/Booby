local dataset = {}

local MACRO = require("MACRO")

local function shuffle(t)
	for i = #t, 2, -1 do
		local j = math.random(i)
		t[i], t[j] = t[j], t[i]
	end
end

local function sep_lines(lines, split_index, train_lines, val_lines)
	for i, line in ipairs(lines) do
		if i <= split_index then
			table.insert(train_lines, line)
		else
			table.insert(val_lines, line)
		end
	end
end

function split(str)
	local result = {}
	local first = true

	for value in string.gmatch(str, "([^,]+)") do
		table.insert(result, first and value or tonumber(value))
		first = false
	end

	return result
end

local function get_min_max(train_lines, train_min, train_max)
	for i, line in ipairs(train_lines) do
		for j, arg in ipairs(line) do
			if j == 1 then goto continue end -- skip M and B

			if not train_min[j] then
				train_min[j] = arg
				train_max[j] = arg
			else
				if arg < train_min[j] then train_min[j] = arg end
				if arg > train_max[j] then train_max[j] = arg end
			end
			::continue::
		end
	end
end

local function arg_scaling(lines, min, max)
	for i, line in ipairs(lines) do
		for j, arg in ipairs(line) do
			if j == 1 then goto continue end

			lines[i][j] = (arg - min[j])/(max[j]-min[j])
			::continue::
		end
	end
end

local function normalize(train_lines, val_lines) -- min max scaling
	local train_min = {}
	local train_max = {}

	for i, line in ipairs(train_lines) do train_lines[i] = split(line) end
	for i, line in ipairs(val_lines) do val_lines[i] = split(line) end

	get_min_max(train_lines, train_min, train_max)

	train_min[1] = 0
	train_max[1] = 0

	local file = assert(io.open("../min_max.lua", "w"), "Failed to create min_max.lua")
	file:write("return {\n")
		file:write("  min = {" .. table.concat(train_min, ", ") .. "},\n")
		file:write("  max = {" .. table.concat(train_max, ", ") .. "},\n")
	file:write("}\n")

	arg_scaling(train_lines, train_min, train_max)
	arg_scaling(val_lines, train_min, train_max)
end

local function write_lines(lines, file, file_name)
	for _, line in ipairs(lines) do
		assert(file:write(line[1]), "Failed to write into "..file_name)

		for j, arg in ipairs(line) do
			if j == 1 then goto continue end

			assert(file:write(","..arg), "Failed to write into "..file_name)
			::continue::
		end
		assert(file:write("\n"), "Failed to write into "..file_name)
	end
end

function dataset.reload()
	os.remove("../data_train.csv")
	os.remove("../data_val.csv")

	local data = assert(io.open("../data.csv", "r"), "Failed to open data.csv")

	local train = assert(io.open("../data_train.csv", "w"), "Failed to create data_train.csv")
	local val = assert(io.open("../data_val.csv", "w"), "Failed to create data_val.csv")

	local M_lines = {}
	local B_lines = {}

	for line in data:lines() do
		local id_index = string.find(line, ",")
		local line_cpy = string.sub(line, id_index + 1)

		local first_char = string.sub(line_cpy, 1, 1)

		if first_char == "M" then
			table.insert(M_lines, line_cpy)
		else
			table.insert(B_lines, line_cpy)
		end
	end
	data:close()

	shuffle(M_lines)
	shuffle(B_lines)

	local split_M = math.floor(#M_lines * MACRO.SPLIT_PERCENT)
	local split_B = math.floor(#B_lines * MACRO.SPLIT_PERCENT)

	local train_lines = {}
	local val_lines = {}

	sep_lines(M_lines, split_M, train_lines, val_lines)
	sep_lines(B_lines, split_B, train_lines, val_lines)

	normalize(train_lines, val_lines) -- min max scaling

	write_lines(train_lines, train, "data_train.csv")
	write_lines(val_lines, val, "data_val.csv")

	train:close()
	val:close()
end

function dataset.force_normalize()
	local train = assert(io.open("../data_train.csv", "r"), "Failed to open data_train.csv")
	local val = assert(io.open("../data_val.csv", "r"), "Failed to open data_val.csv")

	local train_lines = {}
	local val_lines = {}

	for line in train:lines() do
		local id_index = string.find(line, ",")
		local line_cpy = string.sub(line, id_index + 1)
		table.insert(train_lines, line_cpy)
	end

	for line in val:lines() do
		local id_index = string.find(line, ",")
		local line_cpy = string.sub(line, id_index + 1)
		table.insert(val_lines, line_cpy)
	end

	train:close()
	val:close()

	local train = assert(io.open("../data_train.csv", "w"), "Failed to open data_train.csv")
	local val = assert(io.open("../data_val.csv", "w"), "Failed to open data_val.csv")

	normalize(train_lines, val_lines) -- min max scaling

	write_lines(train_lines, train, "data_train.csv")
	write_lines(val_lines, val, "data_val.csv")

	train:close()
	val:close()
end

return dataset
