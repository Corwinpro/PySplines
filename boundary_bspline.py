import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.integrate import simps

class sympy_Bspline():
	def __init__(self, cv, degree = 3, n = 100, periodic = False):
		'''
			Clamped B-Spline with sympy

			cv: list of control points
			degree:   Curve degree
			n: N discretization points
			periodic: default - False; True - Curve is closed
		'''
		self.x = sympy.var('x')

		self.cv = np.array(cv)

		self.periodic = periodic
		self.degree = degree
		self.max_param = self.cv.shape[0] - (self.degree * (1-self.periodic))
		self.kv = self.kv_bspline()
		
		self.construct_bspline_basis()

		self.spline = self.sympy_bspline_expression()
		self.normal = self.sympy_bspline_normal()
		self.sympy_curvature()

		self.T_Point_dict = dict()

		self.normalize_points(n)

	def normalize_points(self, n):

		self.n = 3000
		self.dom = np.linspace(0,self.max_param,self.n)
		self.bspline_getSurface()
		self.n = n

		S_x = sympy.diff(self.spline[0], self.x)
		S_y = sympy.diff(self.spline[1], self.x)

		L = sympy.sqrt(S_x*S_x + S_y*S_y)
		L = self.evaluate_expression(L)

		totalLength = simps(L, self.dom)
		avgDistance =  totalLength / n

		_tmp_dist = 0.
		proper_t_dist = []
		proper_t_dist.append(self.dom[0])

		for i in range(1, len(self.dom) - 1):
			if _tmp_dist < avgDistance:
				_tmp_dist += ( (self.rvals[i][0] - self.rvals[i-1][0])**2. + (self.rvals[i][1] - self.rvals[i-1][1])**2. )**0.5
			else:
				_tmp_dist = 0
				proper_t_dist.append(self.dom[i])
		proper_t_dist.append(self.dom[-1])

		self.dom = proper_t_dist
		self.bspline_getSurface()
		self.n = len(self.dom)

		self.insert_points()

	def dots_angles(self, direction = 'forward'):
		if (direction == 'backward'):
			#Move backwards on surface pointlist
			x,y = np.array(self.rvals[::-1])[:,0], np.array(self.rvals[::-1])[:,1]
		else:
			#Else move as usual
			x,y = np.array(self.rvals)[:,0], np.array(self.rvals)[:,1]

		angles = []
		for i in range(len(x) - 2):
			v1_u = [x[i+1] - x[i], y[i+1] - y[i]]; v1_u /= np.linalg.norm(v1_u)
			v2_u = [x[i+2] - x[i], y[i+2] - y[i]]; v2_u /= np.linalg.norm(v2_u)			
			angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
			angles.append(angle)

		return angles

	def insert_points(self):
		
		tolerance_angle = 0.01

		proper_t_dist = []
		proper_t_dist.append(self.dom[0])
		angles = self.dots_angles()
		tresh_angle = tolerance_angle
		for i in range(len(self.dom) - 2):
			if angles[i]**2. < tresh_angle**2.:
				proper_t_dist.append(self.dom[i+1])
			else:
				n_insert_points = int(angles[i]/tresh_angle)
				dt = (self.dom[i+2] - self.dom[i+1])/n_insert_points
				for j in range(n_insert_points):
					proper_t_dist.append(self.dom[i+1] + j*dt)
		proper_t_dist.append(self.dom[-1])
		self.dom = proper_t_dist
		self.bspline_getSurface()
		self.n = len(self.dom)
		
		proper_t_dist = []
		proper_t_dist.append(self.dom[-1])
		angles = self.dots_angles(direction = 'backward')[::-1]
		tresh_angle = tolerance_angle
		for i in range(len(self.dom) - 2):
			k = -1-i
			if angles[k]**2. < tresh_angle**2.:
				proper_t_dist.append(self.dom[k-1])
			else:
				n_insert_points = int(angles[k]/tresh_angle)
				dt = (self.dom[k-2] - self.dom[k-1])/n_insert_points
				for j in range(n_insert_points):
					proper_t_dist.append(self.dom[k-1] + j*dt)#-j
		proper_t_dist.append(self.dom[0])
		self.dom = proper_t_dist[::-1]
		self.bspline_getSurface()
		self.n = len(self.dom)
		
	def kv_bspline(self):
	    """ 
	    	Calculate knot vector of a bspline
	    """
	    cv = np.asarray(self.cv)
	    count = cv.shape[0]

	    # Closed curve
	    if self.periodic:
	        kv = np.arange(-self.degree,count+self.degree+1)
	        factor, fraction = divmod(count+self.degree+1, count)
	        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
	        cv = cv[:-1]
	        degree = np.clip(self.degree,1,self.degree)
	    # Opened curve
	    else:
	        degree = np.clip(self.degree,1,count-1)
	        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

	    self.cv = cv
	    return list(kv)

	def evaluate_expression(self, expression, point = None):

		expression_val = []
		if (isinstance(expression, list)):
			n = len(expression)
			#lambda_expression = [sympy.lambdify(self.x, expression[i]) for i in range(n)]
			if (point is not None):
				vals = []
				for i in range(n):
					vals.append(float(expression[i].subs(self.x, point)))#HERE
				return vals
			else:
				lambda_expression = [sympy.lambdify(self.x, expression[i]) for i in range(n)]
				for r in self.dom:
					vals = []
					for i in range(n):
						vals.append(lambda_expression[i](r))
					expression_val.append(vals)
		else:
			#lambda_expression = sympy.lambdify(self.x, expression)
			if (point is not None):
				return float(expression.subs(self.x, point))
			else:
				lambda_expression = sympy.lambdify(self.x, expression)				
				for r in self.dom:
					val = lambda_expression(r)
					expression_val.append(float(val))

		return expression_val#np.array(expression_val)

	def construct_bspline_basis(self):
		self.bspline_basis = []
		for i in range(len(self.cv)):
			self.bspline_basis.append(sympy.bspline_basis(self.degree, self.kv, i, self.x))
		self.lambdify_bspline_basis_expressions()

	def lambdify_bspline_basis_expressions(self):

		self.bspline_basis_lambda = []
		for i in range(len(self.cv)):
			self.bspline_basis_lambda.append(sympy.lambdify(self.x, self.bspline_basis[i]))

	def mass_matrix(self, reduced = False):
		M = []
		S_x = sympy.diff(self.spline[0], self.x)
		S_y = sympy.diff(self.spline[1], self.x)

		L = sympy.sqrt(S_x*S_x + S_y*S_y)
		L = self.evaluate_expression(L)

		if (reduced):
			index_range = range(1, len(self.cv) - 1)
		else:
			index_range = range(len(self.cv))

		for i in index_range:
			M_i = []
			bfunction1 = np.array(self.evaluate_expression(self.bspline_basis[i]))
			for j in index_range:#range(len(self.cv)):
				bfunction2 = np.array(self.evaluate_expression(self.bspline_basis[j]))
				M_ij = simps(bfunction1*bfunction2*L, self.dom)
				M_i.append(M_ij)
			M.append(M_i)
			#print M_i, '\n'
		#M_inv = np.linalg.inv(M)
		#print M_inv
		return M#_inv

	def DLMM_mass_matrix(self):
		M = []
		S_x = sympy.diff(self.spline[0], self.x)
		S_y = sympy.diff(self.spline[1], self.x)

		L = sympy.sqrt(S_x*S_x + S_y*S_y)
		L = self.evaluate_expression(L)
		for i in range(len(self.cv)):
			M_i = []
			for j in range(len(self.cv)):
				if (i == j):
					bfunction = np.array(self.evaluate_expression(self.bspline_basis[i]))
					M_ij = simps(bfunction*L, self.dom)				
				else:
					M_ij = 0.
				M_i.append(M_ij)
			M.append(M_i)
			#print M_i, '\n'
		M_inv = np.linalg.inv(M)

		return M_inv		

	def construct_bspline_basis_derivative(self):
		self.bspline_derivative_basis = []
		for i in range(len(self.cv)):
			self.bspline_derivative_basis.append(sympy.diff(self.bspline_basis[i], self.x))

	def sympy_bspline_expression(self):
		bspline_expression = [0, 0]
		
		for i in range(len(self.cv)):
			bspline_expression[0] += self.cv[i][0]*self.bspline_basis[i]
			bspline_expression[1] += self.cv[i][1]*self.bspline_basis[i]
		return bspline_expression

	def sympy_bspline_normal(self):
		S_x = sympy.diff(self.spline[0], self.x)
		S_y = sympy.diff(self.spline[1], self.x)

		return [-S_y/sympy.sqrt(S_x*S_x + S_y*S_y) , S_x/sympy.sqrt(S_x*S_x + S_y*S_y)]
	
	def sympy_curvature(self):

		S_x = sympy.diff(self.spline[0], self.x)
		S_y = sympy.diff(self.spline[1], self.x)

		S_xx = sympy.diff(self.spline[0], self.x, 2)
		S_yy = sympy.diff(self.spline[1], self.x, 2)

		self.curvature =  (S_x*S_yy - S_y*S_xx)/sympy.sqrt(S_x*S_x + S_y*S_y)**1.5

		self.lambdify_curvature_expression()

	def bspline_getSurface(self):

		self.rvals = self.evaluate_expression(self.spline)

		for i in range(len(self.rvals)):
			self.T_Point_dict[self.dom[i]] = self.rvals[i]
			self.rvals[i][0] = int(self.rvals[i][0]*1.e8)/1.e8
			self.rvals[i][1] = int(self.rvals[i][1]*1.e8)/1.e8

	def bspline_getNormal(self):

		self.normalField = self.evaluate_expression(self.normal)

	def bspline_getCurvature(self):

		self.curvatureField = self.evaluate_expression(self.curvature)
		self.curvatureField = [item for sublist in self.curvatureField for item in sublist]

	def surface_area(self):

		y = np.array(self.rvals)[:,1]
		from time import time
		x_t = self.evaluate_expression(sympy.diff(self.spline[0], self.x))
		surface_area = simps(y*x_t, self.dom)
		return surface_area

	def surface_area_gradient(self):

		'''
		Linear gradient of surface wrt positions of control points
		'''
		self.construct_bspline_basis_derivative()

		N = np.zeros((len(self.cv),len(self.cv)))
		bspline = [np.array(self.evaluate_expression(self.bspline_basis[i])) for i in range(len(self.cv))]
		bspline_derivative = [np.array(self.evaluate_expression(self.bspline_derivative_basis[i])) for i in range(len(self.cv))]
		
		for i in range(len(self.cv)):
			Ni = bspline[i]
			for j in range(len(self.cv)):
				Nj = bspline_derivative[j]
				N[i][j] = simps(Ni*Nj, self.dom)

		grad = []
		for i in range(len(self.cv)):
			grad_x = grad_y = 0.
			for j in range(len(self.cv)):
				grad_x += self.cv[j][1]*N[j][i]
				grad_y += self.cv[j][0]*N[i][j]
			grad.append([grad_x, grad_y])

		return grad

	def get_t_from_point(self, point):
		
		point = list(point)
		for key, value in self.T_Point_dict.items():
			if value == point:
				return key

		self.tolerance = 1.e-8
		min_dist = previous_distance = 1.e8
		
		for r in self.rvals:
			current_distance = np.linalg.norm(np.array(r)-np.array(point))
			if (current_distance > previous_distance):
				continue
			if (current_distance < min_dist):
				min_dist = current_distance
				closest_point = r
			previous_distance = current_distance

		index = self.rvals.index(closest_point)
		t = self.dom[index]

		#If we are 'very' close to an existing point, we just return t
		#This helps if a point is 'out of the domain' but still very close
		if (min_dist < self.tolerance):
			return t
		#We get the distance to the points to the left and to the right of the closest point
		# ...---o---o---left---closest---right---o----o---...
		#If the first point of the spline is the nearest, there are no points to the left
		#And the second nearest point (to the right) has the second shortest distance
		#points_distance is the distance between to bspline points closest to the point
		#closest---right---o----o---...
		if (index == 0):
			second_distance = np.linalg.norm(np.array(self.rvals[1])-np.array(point))
			points_distance = np.linalg.norm(np.array(self.rvals[1])-np.array(self.rvals[0]))
			second_t = self.dom[1]
		#Same happens if the closest point is the last one on the spline
		#...---o---left---closest.
		elif (index == len(self.dom)-1):
			second_distance = np.linalg.norm(np.array(self.rvals[-2])-np.array(point))
			points_distance = np.linalg.norm(np.array(self.rvals[-1])-np.array(self.rvals[-2]))
			second_t = self.dom[-2]
		#Otherwise, the closest bspline point to the point is somewhere on the curve
		else:
			left_point_distance = np.linalg.norm(np.array(self.rvals[index-1])-np.array(point))
			right_point_distance = np.linalg.norm(np.array(self.rvals[index+1])-np.array(point))
		
			r_right = [self.rvals[index+1][0] - point[0], self.rvals[index+1][1] - point[1]]
			r_left  = [self.rvals[index-1][0] - point[0], self.rvals[index-1][1] - point[1]]
			r_min =   [self.rvals[index][0] - point[0], self.rvals[index][1] - point[1]]
			if (np.dot(r_right, r_min) > 0):
				second_distance = left_point_distance
				second_t = self.dom[index-1]
				points_distance = np.linalg.norm(np.array(self.rvals[index-1])-np.array(closest_point))
			else:
				second_distance = right_point_distance
				second_t = self.dom[index+1]
				points_distance = np.linalg.norm(np.array(self.rvals[index+1])-np.array(closest_point))

		if (np.fabs(second_distance + min_dist - points_distance) > self.tolerance):
			not_online_error_str = str('The point ' + str(point) + ' is not on the lines segments, tolerance violated by ' + 
							str(np.fabs(second_distance + min_dist - points_distance)/self.tolerance) + 
							' times.')
			raise ValueError(not_online_error_str)
		else:
			t_interpolated = t*second_distance/points_distance + second_t*min_dist/points_distance

		self.T_Point_dict[t_interpolated] = point

		return t_interpolated

	def get_normal_from_point(self, point):

		t = self.get_t_from_point(point)
		
		return self.evaluate_expression(self.normal, point = t) 

	def get_curvature_from_point(self,point):

		if self.degree == 1:
			return 0.
			
		t = self.get_t_from_point(point)

		return self.bspline_curvature_lambda(t).item()
		#return self.evaluate_expression(self.curvature, point = t)

	def get_displacement_from_point(self, point, controlPointNumber):

		t = self.get_t_from_point(point)
		#displacement = self.evaluate_expression(self.bspline_basis[controlPointNumber], point= t).item()
		displacement = self.bspline_basis_lambda[controlPointNumber](t).item()
		return [displacement, displacement]

	def lambdify_curvature_expression(self):
		
		self.bspline_curvature_lambda = sympy.lambdify(self.x, self.curvature)

	def plot(self, linetype = '-', window = plt):
		
		window.plot(np.array(self.rvals)[:,0], np.array(self.rvals)[:,1], linetype, color = 'black')
		self.plot_cv(window)

	def plot_cv(self, window = plt):
		
		window.plot(self.cv[:,0], self.cv[:,1], 'o', markersize = 4, c = 'black', mfc='none')

	def plot_normals(self):
		
		plt.plot(self.rvals[:,0] + self.normalField[:,0], self.rvals[:,1] + self.normalField[:,1], ',-')

	def plot_curvature(self):

		from mpl_toolkits.mplot3d import Axes3D
		self.bspline_getCurvature()
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot(np.array(self.rvals)[:,0][::5], np.array(self.rvals)[:,1][::5], self.curvatureField[::5])
		plt.show()

	def plot_all(self, linetype = '-', grid = True):
		self.plot(linetype = linetype)
		#self.plot_cv()
		#self.plot_normals()
		#plt.grid(grid)

	def plot_to_pdf(self, pdfname = 'bspline_boundary'):
	
		plt.savefig(pdfname, bbox_inches='tight')

class Example_BSpline(sympy_Bspline):
	'''
		Example BSpline with degree d = 3, n = 100 plotted points, default periodic = False
	'''	
	def __init__(self, degree = 2, n = 100, periodic = False):
		top_control_points = [[0.,0.0], [0.1,0.0], [0.2,0.0], [0.3,0.0], [0.4,0.0]]
		#for i in range(15):
		#	top_control_points += [[2.2+i, 0.7]]
		#top_control_points += [[16.3, 0.7], [16.4, 0.7], [16.4, 0.8], [16.4, 1.7], [16.4, 2.7], [16.4, 3.7], [16.4, 4.7]]
		self.example_cv = top_control_points
		#self.example_cv = [[0.0,0.1], [0.2, 0.1], [0.4, 0.05], [0.6, 0.05], [0.8,0.1],[1.0,0.1]]
		self.cv = np.array(self.example_cv)
		sympy_Bspline.__init__(self, self.cv, degree = degree, n = n, periodic = periodic)
		#self.plot_example()

	def plot_example(self):
		self.bspline_getNormal()
		self.plot_all()

if __name__ == "__main__":

	pass