local MACRO = require("MACRO")
local dataset = require("dataset")
local parsing = require("parsing")
local ai = require("ai")

local args = {...}

function check_for_reload()
	if MACRO.RELOAD_DATASET or not io.open("../data_train.csv", "r") or not io.open("../data_val.csv", "r") then
		dataset.reload()
		os.exit()
	end
end

function check_for_training()
	if MACRO.TRAIN or not io.open("../".."MODEL_"..MACRO.HIDED_LAYER.."L_"..MACRO.HIDED_LAYER_SIZE.."LS_"..MACRO.LEARNING_RATE.."LR_"..MACRO.EPOCH.."E"..".lua", "r") then
		ai.start_train()
	end
end

function main()
	parsing.parse_args(args)

	if MACRO.PREDICT then
		ai.predict(MACRO.PREDICT)
		return
	end

	check_for_reload()
	check_for_training()
end

main()
