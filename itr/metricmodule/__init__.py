def second2DHM(seconds):
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	strtime = f'{round(hours)}:{str(round(minutes)).zfill(2)}:{str(round(seconds)).zfill(2)}'
	return strtime, hours, minutes, seconds
