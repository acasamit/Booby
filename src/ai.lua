local ai = {}

local MACRO = require("MACRO")

local layers = {}
local history = {loss = {}, f1 = {}, epochs = {}, precision = {}, recall = {}}

local function split(str)
	local result = {}
	for value in string.gmatch(str, "([^,]+)") do
		table.insert(result, tonumber(value))
	end
	return result
end

local function softmax(output_layer)
	local max_z = -math.huge

	for _, n in ipairs(output_layer) do
		if n.z > max_z then max_z = n.z end
	end

	local sum = 0
	local exps = {}

	for i, n in ipairs(output_layer) do
		exps[i] = math.exp(n.z - max_z)
		sum = sum + exps[i]
	end

	for i, n in ipairs(output_layer) do
		n.activation = exps[i] / sum
	end
end

local function xavier_init(n_in, n_out)
	local limit = math.sqrt(6 / (n_in + n_out))
	return (math.random() * 2 - 1) * limit
end

function setup_neural_network()
	for i = 1, MACRO.HIDED_LAYER, 1 do
		layers[i] = {}

		for j = 1, MACRO.HIDED_LAYER_SIZE, 1 do
			table.insert(layers[i], {
				z = 0,
				weight = {},
				m_weight = {},
				v_weight = {},
				biase = 0,
				m_bias = 0,
				v_bias = 0,
				acc_weight_grad = {},
				acc_bias_grad = 0
			})

			local size = i == 1 and MACRO.INPUT_LAYER or MACRO.HIDED_LAYER_SIZE
			for k = 1, size, 1 do
				table.insert(layers[i][j].weight, xavier_init(size, MACRO.HIDED_LAYER_SIZE))
				table.insert(layers[i][j].m_weight, 0)
				table.insert(layers[i][j].v_weight, 0)
				table.insert(layers[i][j].acc_weight_grad, 0)
			end
		end
	end

	layers[#layers + 1] = {}
	local output_neurons

	for i = 1, MACRO.OUTPUT_LAYER, 1 do
		table.insert(layers[#layers], {
			z = 0,
			weight = {},
			m_weight = {},
			v_weight = {},
			biase = 0,
			m_bias = 0,
			v_bias = 0,
			acc_weight_grad = {},
			acc_bias_grad = 0
		})

		for j = 1, MACRO.HIDED_LAYER_SIZE, 1 do
			table.insert(layers[#layers][i].weight, xavier_init(MACRO.HIDED_LAYER_SIZE, MACRO.OUTPUT_LAYER))
			table.insert(layers[#layers][i].m_weight, 0)
			table.insert(layers[#layers][i].v_weight, 0)
			table.insert(layers[#layers][i].acc_weight_grad, 0)
		end
	end
end

local function feedforward(input, column)
	for i = 1, #layers[column], 1 do
		layers[column][i].z = 0

		for j, val in ipairs(input) do
			layers[column][i].z = layers[column][i].z + val * layers[column][i].weight[j]
		end

		layers[column][i].z = layers[column][i].z + layers[column][i].biase

		if column ~= #layers then
			layers[column][i].activation = math.max(0, layers[column][i].z)
		end
	end
end

local function update_weights(batch_size)
	for i = 1, #layers do
		for j = 1, #layers[i] do
			local n = layers[i][j]

			-- weight
			for k = 1, #n.weight do
				local grad = n.acc_weight_grad[k] / batch_size
				
				n.m_weight[k] = MACRO.BETA1 * n.m_weight[k] + (1 - MACRO.BETA1) * grad
				n.v_weight[k] = MACRO.BETA2 * n.v_weight[k] + (1 - MACRO.BETA2) * (grad^2)
				
				local mh = n.m_weight[k] / (1 - MACRO.BETA1^MACRO.LINE_READ)
				local vh = n.v_weight[k] / (1 - MACRO.BETA2^MACRO.LINE_READ)
				
				n.weight[k] = n.weight[k] - MACRO.LEARNING_RATE * (mh / (math.sqrt(vh) + MACRO.EPSILON))
				n.acc_weight_grad[k] = 0
			end

			-- bias
			local b_grad = n.acc_bias_grad / batch_size
			n.m_bias = MACRO.BETA1 * n.m_bias + (1 - MACRO.BETA1) * b_grad
			n.v_bias = MACRO.BETA2 * n.v_bias + (1 - MACRO.BETA2) * (b_grad^2)
			
			local mbh = n.m_bias / (1 - MACRO.BETA1^MACRO.LINE_READ)
			local vbh = n.v_bias / (1 - MACRO.BETA2^MACRO.LINE_READ)
			
			n.biase = n.biase - MACRO.LEARNING_RATE * (mbh / (math.sqrt(vbh) + MACRO.EPSILON))
			n.acc_bias_grad = 0
		end
	end
end

local function backdrop(Loss, P, result, input)
	for i = #layers, 1, -1 do -- layer

		for j = 1, #layers[i], 1 do -- neuron
			if i ~= #layers then
				local sum = 0

				for n = 1, #layers[i + 1] do
					sum = sum + (layers[i + 1][n].delta * layers[i + 1][n].weight[j])
				end

				layers[i][j].delta = sum * (layers[i][j].z > 0 and 1 or 0)
			else
				local target = (j == 1) and result or (1 - result)
				layers[i][j].delta = P[j] - target
			end

			layers[i][j].acc_bias_grad = layers[i][j].acc_bias_grad + layers[i][j].delta

			for k = 1, #layers[i][j].weight, 1 do -- weight
				local g = layers[i][j].delta * (i == 1 and input[k] or layers[i-1][k].activation)
				layers[i][j].acc_weight_grad[k] = layers[i][j].acc_weight_grad[k] + g
			end
		end
	end
end

local function epoch()
	local train = assert(io.open("../data_train.csv", "r"), "Failed to open data_train.csv")
	local loss_sum, epoch_n, count = 0, 0, 0
	local TP, TN, FP, FN = 0, 0, 0, 0

	for line in train:lines() do
		local r_index = string.find(line, ",")
		local input = split(string.sub(line, r_index + 1))
		local first_input = input
		local result = (string.sub(line, 1, 1) == "B") and 0 or 1

		for column = 1, #layers do
			feedforward(input, column)

			input = {}
			for i = 1, #layers[column] do
				table.insert(input, layers[column][i].activation or layers[column][i].z)
			end
		end

		softmax(layers[#layers])

		local P = {}

		for n = 1, #layers[#layers] do table.insert(P, layers[#layers][n].activation) end
		local Loss = -((result * math.log(math.max(P[1], 1e-15))) + (1 - result) * math.log(math.max(1 - P[1], 1e-15)))
		
		loss_sum = loss_sum + Loss
		epoch_n = epoch_n + 1

		local prediction = P[1] > 0.5 and 1 or 0

		if prediction == 1 and result == 1 then TP = TP + 1
		elseif prediction == 0 and result == 0 then TN = TN + 1
		elseif prediction == 1 and result == 0 then FP = FP + 1
		elseif prediction == 0 and result == 1 then FN = FN + 1 end

		backdrop(Loss, P, result, first_input)
		count = count + 1

		if count >= MACRO.BATCH_SIZE then
			MACRO.LINE_READ = MACRO.LINE_READ + 1
			update_weights(count)
			count = 0
		end
	end

	if count > 0 then
		MACRO.LINE_READ = MACRO.LINE_READ + 1
		update_weights(count)
	end

	local final_loss = loss_sum / epoch_n
	local precision = (TP + FP == 0) and 0 or TP / (TP + FP)
	local recall = (TP + FN == 0) and 0 or TP / (TP + FN)
	local F1 = (precision + recall == 0) and 0 or 2 * (precision * recall) / (precision + recall)

	print("(Train) L: "..final_loss.." P: "..precision.." R: "..recall.." F1: "..F1)
	train:close()
end

function shuffle_file(input_path, output_path)
	local f = assert(io.open(input_path, "r"), "Failed to open input file")

	local lines = {}
	for line in f:lines() do
		table.insert(lines, line)
	end
	f:close()

	for i = #lines, 2, -1 do
		local j = math.random(i)
		lines[i], lines[j] = lines[j], lines[i]
	end

	local out = assert(io.open(output_path, "w"), "Failed to open output file")
	for _, line in ipairs(lines) do
		out:write(line .. "\n")
	end
	out:close()
end

function ai.test()
	local val = assert(io.open("../data_val.csv", "r"), "Failed to open data_val.csv")
	local loss_sum, epoch_n = 0, 0
	local TP, TN, FP, FN = 0, 0, 0, 0

	for line in val:lines() do
		local r_index = string.find(line, ",")
		local input = split(string.sub(line, r_index + 1))
		local result = (string.sub(line, 1, 1) == "B") and 0 or 1

		for column = 1, #layers do
			feedforward(input, column)

			input = {}
			for i = 1, #layers[column] do
				table.insert(input, layers[column][i].activation or layers[column][i].z)
			end
		end

		softmax(layers[#layers])

		local P = {}

		for n = 1, #layers[#layers] do table.insert(P, layers[#layers][n].activation) end
		local Loss = -((result * math.log(math.max(P[1], 1e-15))) + (1 - result) * math.log(math.max(1 - P[1], 1e-15)))
		
		loss_sum = loss_sum + Loss
		epoch_n = epoch_n + 1

		local prediction = P[1] > 0.5 and 1 or 0

		if prediction == 1 and result == 1 then TP = TP + 1
		elseif prediction == 0 and result == 0 then TN = TN + 1
		elseif prediction == 1 and result == 0 then FP = FP + 1
		elseif prediction == 0 and result == 1 then FN = FN + 1 end
	end

	local final_loss = loss_sum / epoch_n
	local precision = (TP + FP == 0) and 0 or TP / (TP + FP)
	local recall = (TP + FN == 0) and 0 or TP / (TP + FN)
	local F1 = (precision + recall == 0) and 0 or 2 * (precision * recall) / (precision + recall)

	table.insert(history.loss, final_loss)
	table.insert(history.precision, precision)
	table.insert(history.recall, recall)
	table.insert(history.f1, F1)
	table.insert(history.epochs, #history.loss)

	print("(Val)   L: "..final_loss.." P: "..precision.." R: "..recall.." F1: "..F1)
	val:close()

	return F1
end

function save_model(filename)
	local file = assert(io.open(filename, "w"), "Failed so save best model (cannot create "..filename..")")
	file:write("return {\n")
	for i = 1, #layers do
		file:write("  {\n")
		for j = 1, #layers[i] do
			local n = layers[i][j]
			file:write("    {biase = " .. n.biase .. ", weight = {")
			for k = 1, #n.weight do
				file:write(n.weight[k] .. (k == #n.weight and "" or ", "))
			end
			file:write("}},\n")
		end
		file:write("  },\n")
	end
	file:write("}\n")
	file:close()
end

local function serialize_array(t)
	local out = {}
	for i = 1, #t do
		out[i] = tostring(t[i])
	end
	return table.concat(out, ", ")
end

function save_run_data(run_name)
	local path = "../graph/data.js"
	local file_exists = false

	do
		local f = io.open(path, "r")
		if f then
			file_exists = true
			f:close()
		end
	end

	local file = assert(io.open(path, "a"))

	if not file_exists or io.open(path, "r"):seek("end") == 0 then
		file:write("window.runs = window.runs || {};\n")
	end

	file:write('window.runs["' .. run_name .. '"] = {\n')
	file:write("  labels: [" .. serialize_array(history.epochs) .. "],\n")
	file:write("  loss: [" .. serialize_array(history.loss) .. "],\n")
	file:write("  precision: [" .. serialize_array(history.precision) .. "],\n")
	file:write("  recall: [" .. serialize_array(history.recall) .. "],\n")
	file:write("  f1: [" .. serialize_array(history.f1) .. "]\n")
	file:write("};\n")

	file:close()
end

function load_model(filename)
	local data = dofile(filename)

	layers = {}
	for i = 1, #data do
		layers[i] = {}
		for j = 1, #data[i] do
			layers[i][j] = {
				biase = data[i][j].biase,
				weight = data[i][j].weight,
				z = 0,
				activation = 0,

				m_weight = {}, v_weight = {},
				m_bias = 0, v_bias = 0
			}
			for k = 1, #layers[i][j].weight do
				layers[i][j].m_weight[k] = 0
				layers[i][j].v_weight[k] = 0
			end
		end
	end
end

function ai.start_train()
	setup_neural_network()

	history = {loss = {}, f1 = {}, epochs = {}, precision = {}, recall = {}}
	local best_f1 = 0
	local last_best = 0

	for i = 1, MACRO.EPOCH, 1 do
		print("Epoch "..i.."/"..MACRO.EPOCH..":")
		MACRO.EPOCH_INDEX = i

		epoch()

		local current_f1 = ai.test()
		if current_f1 > best_f1 then
			best_f1 = current_f1
			last_best = i

			save_model("../".."MODEL_"..MACRO.HIDED_LAYER.."L_"..MACRO.HIDED_LAYER_SIZE.."LS_"..MACRO.LEARNING_RATE.."LR_"..MACRO.EPOCH.."E".."_B"..MACRO.BATCH_SIZE..".lua")
		end

		if MACRO.EARLY_STOPPING > 0 and MACRO.EPOCH_INDEX - last_best >= MACRO.EARLY_STOPPING then
			print("Eartly Stopping the training")
			break
		end

		shuffle_file("../data_train.csv", "../data_train.csv")
	end

	save_run_data("MODEL_"..MACRO.HIDED_LAYER.."L_"..MACRO.HIDED_LAYER_SIZE.."LS_"..MACRO.LEARNING_RATE.."LR_"..MACRO.EPOCH.."E".."_B"..MACRO.BATCH_SIZE)
	print("Best Model F1: "..best_f1)
end

function ai.predict(filename)
	local file = assert(io.open(filename, "r"), "Failed to open "..filename)
	print("Using file "..filename.."...")

	load_model(MACRO.MODEL_USED)
	print("Using model "..MACRO.MODEL_USED.."...")

	for line in file:lines() do
		local input = split(line)

		for column = 1, #layers do
			feedforward(input, column)

			input = {}
			for i = 1, #layers[column] do
				table.insert(input, layers[column][i].activation or layers[column][i].z)
			end
		end

		softmax(layers[#layers])
		local P = layers[#layers][1].activation

		if P > 0.5 then
			print("M: "..string.format("%.3f", P))
		else
			print("B: ".. string.format("%.3f", 1 - P))
		end
	end
end

return ai
