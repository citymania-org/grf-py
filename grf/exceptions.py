class FormatError(Exception):
	def __init__(self, obj, message):
		self.obj = obj
		super().__init__(message)
