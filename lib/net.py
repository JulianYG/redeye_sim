
class NetTree:

	def __init__(self, name, layerInfo, top, bottom):
		self.depth = 0
		self.name = name
		self.layer = layerInfo
		self.top = top
		self.bottom = bottom

	def get_name(self):
		return self.name

	def set_name(self, name):
		self.name = name

	def get_depth(self):
		return self.depth

	def set_depth(self, d):
		self.depth = d

	def get_parent(self):
		return self.parent

	def get_bottom(self):
		return self.bottom

	def get_top(self):
		return self.top

	def add_bottom(self, bottom):
		self.bottom.append(bottom)

	def add_top(self, top):
		self.top.append(top)

	def set_bottom(self, bottom):
		self.bottom = bottom

	def set_top(self, top):
		self.top = top

	def get_type(self):
		return self.layer.type

	def insert_layer(self, node):
		"""
		Insert node right after current head
		"""
		for b_l in self.bottom:
			b_l.set_bottom([node])
			node.add_top(b_l)
		self.set_top([node])
		node.add_bottom(self)


def construct_tree(tree, layers):
	for l in layers:
		

# class NetLeaf:

# 	def __init__(self, name, layerInfo, bottom):
# 		self.name = name
# 		self.layer = layerInfo

# 	def get_type(self):
# 		return self.layer.type

# 	def get_name(self):
# 		return self.name

# 	def get_depth(self):
# 		return 0

# 	def get_bottom(self):

# 	def get_top(self):
		

