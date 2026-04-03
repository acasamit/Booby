local MACRO = require("MACRO")
local dataset = require("dataset")
local parsing = require("parsing")
local ai = require("ai")

local args = {...}

function check_for_reload()
	if MACRO.RELOAD_DATASET or not io.open("../data_train.csv", "r") or not io.open("../data_val.csv", "r") then
		dataset.reload()

		if MACRO.RELOAD_DATASET then
			os.exit()
		end
	end
end

function check_for_training()
	if MACRO.TRAIN or not io.open("../".."MODEL_"..MACRO.HIDED_LAYER.."L_"..MACRO.HIDED_LAYER_SIZE.."LS_"..MACRO.LEARNING_RATE.."LR_"..MACRO.EPOCH.."E".."_B"..MACRO.BATCH_SIZE..".lua", "r") then
		ai.start_train()
	else
		print("There is already a model with this name, if you want to override it use -t")
	end
end

function main()
	math.randomseed(os.time())

	parsing.parse_args(args)

	if MACRO.FORCE_NORMALIZE then
		dataset.force_normalize()
	end

	if MACRO.PREDICT then
		ai.predict(MACRO.PREDICT)
		return
	end

	check_for_reload()
	check_for_training()
end

main()
