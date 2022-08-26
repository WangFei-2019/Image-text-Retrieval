from . import Models


def get_model(config):
	if config['name'] == 'VSE++':
		model = Models.VSE_PP(config)
	elif config['name'] == 'SCAN':
		model = Models.SCAN(config)
	elif config['name'] == 'VSRN':
		model = Models.VSRN(config)
	elif config['name'] == 'SAEM':
		model = Models.SAEM(config)
	elif config['name'] == 'SGRAF':
		model = Models.SGRAF(config)
	elif config['name'] == 'CAMERA':
		model = Models.CAMERA(config)
	else:
		raise KeyError(f'No model is named {config["name"]}')
	return model
