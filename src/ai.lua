local ai = {}

local MACRO = require("MACRO")

local layers = {}

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

	print(loss_sum / epoch_n)
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

function ai.start_train()
	setup_neural_network()

	for i = 1, MACRO.EPOCH, 1 do
		print("Epoch "..i.."/"..MACRO.EPOCH..":")
		MACRO.EPOCH_INDEX = i

		epoch()
		shuffle_file("../data_train.csv", "../data_train.csv")
	end
end

return ai
