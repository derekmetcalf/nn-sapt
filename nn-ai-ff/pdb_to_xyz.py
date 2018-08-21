import argparse

class atom:
	def __init__(self, read):
		listo=read.split()
		self.type=read[77]
		self.x=read[31:38]
		self.y=read[40:46]
		self.z=read[47:54]
		k=self.type+' atom added'
		print(k)

class molecule:
	def __init__(self, molfile):
		self.Molfile=open(molfile, 'r')
		#self.basis=self.Molfile.readline().rstrip('\n')
		self.atoms=[]
		self.basis_functions=[]
		self.energy=0.0
		i=0
		for line in self.Molfile:
			if line.split()[0] is 'ATOM' or 'HETATM':
				self.atoms.append(atom(line))
				i=i+1
				print(i)
			else:
				print(line)	

	def xyz_print(self, xyz_name):
		xyz_scribe=open(xyz_name, 'w')
		xyz_scribe.write(str(len(self.atoms))+'\n')
		for atom in self.atoms:
			xyz_scribe.write(atom.type+'\t'+ atom.x+'\t'+atom.y+'\t'+'\t'+atom.z+'\n')
