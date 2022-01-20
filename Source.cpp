#include "mvector.h"
#include "mmatrix.h"

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>


////////////////////////////////////////////////////////////////////////////////
// Set up random number generation

// Set up a "random device" that generates a new random number each time the program is run
std::random_device rand_dev;

// Set up a pseudo-random number generater "rnd", seeded with a random number
std::mt19937 rnd(rand_dev());

// Alternative: set up the generator with an arbitrary constant integer. This can be useful for
// debugging because the program produces the same sequence of random numbers each time it runs.
// To get this behaviour, uncomment the line below and comment the declaration of "rnd" above.
//std::mt19937 rnd(12345);


////////////////////////////////////////////////////////////////////////////////
// Some operator overloads to allow arithmetic with MMatrix and MVector.
// These may be useful in helping write the equations for the neural network in
// vector form without having to loop over components manually. 
//
// You may not need to use all of these; conversely, you may wish to add some
// more overloads.

// MMatrix * MVector
MVector operator*(const MMatrix& m, const MVector& v)
{
	assert(m.Cols() == v.size());

	MVector r(m.Rows());

	for (int i = 0; i < m.Rows(); i++)
	{
		for (int j = 0; j < m.Cols(); j++)
		{
			r[i] += m(i, j) * v[j];
		}
	}
	return r;
}

// transpose(MMatrix) * MVector
MVector TransposeTimes(const MMatrix& m, const MVector& v)
{
	assert(m.Rows() == v.size());

	MVector r(m.Cols());

	for (int i = 0; i < m.Cols(); i++)
	{
		for (int j = 0; j < m.Rows(); j++)
		{
			r[i] += m(j, i) * v[j];
		}
	}
	return r;
}

// MVector + MVector
MVector operator+(const MVector& lhs, const MVector& rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i = 0; i < lhs.size(); i++)
		r[i] += rhs[i];

	return r;
}

// MVector - MVector
MVector operator-(const MVector& lhs, const MVector& rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i = 0; i < lhs.size(); i++)
		r[i] -= rhs[i];

	return r;
}

// MMatrix = MVector <outer product> MVector
// M = a <outer product> b
MMatrix OuterProduct(const MVector& a, const MVector& b)
{
	MMatrix m(a.size(), b.size());
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++)
		{
			m(i, j) = a[i] * b[j];
		}
	}
	return m;
}

// Hadamard ("a-da-MAR") product
MVector operator*(const MVector& a, const MVector& b)
{
	assert(a.size() == b.size());

	MVector r(a.size());
	for (int i = 0; i < a.size(); i++)
		r[i] = a[i] * b[i];
	return r;
}

// double * MMatrix
MMatrix operator*(double d, const MMatrix& m)
{
	MMatrix r(m);
	for (int i = 0; i < m.Rows(); i++)
		for (int j = 0; j < m.Cols(); j++)
			r(i, j) *= d;

	return r;
}

// double * MVector
MVector operator*(double d, const MVector& v)
{
	MVector r(v);
	for (int i = 0; i < v.size(); i++)
		r[i] *= d;

	return r;
}

// MVector -= MVector
MVector operator-=(MVector& v1, const MVector& v)
{
	assert(v1.size() == v.size());

	for (int i = 0; i < v1.size(); i++)
		v1[i] -= v[i];

	return v1;
}

// MMatrix -= MMatrix
MMatrix operator-=(MMatrix& m1, const MMatrix& m2)
{
	assert(m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols());

	for (int i = 0; i < m1.Rows(); i++)
		for (int j = 0; j < m1.Cols(); j++)
			m1(i, j) -= m2(i, j);

	return m1;
}

// Output function for MVector
inline std::ostream& operator<<(std::ostream& os, const MVector& rhs)
{
	std::size_t n = rhs.size();
	os << "(";
	for (std::size_t i = 0; i < n; i++)
	{
		os << rhs[i];
		if (i != (n - 1)) os << ", ";
	}
	os << ")";
	return os;
}

// Output function for MMatrix
inline std::ostream& operator<<(std::ostream& os, const MMatrix& a)
{
	int c = a.Cols(), r = a.Rows();
	for (int i = 0; i < r; i++)
	{
		os << "(";
		for (int j = 0; j < c; j++)
		{
			os.width(10);
			os << a(i, j);
			os << ((j == c - 1) ? ')' : ',');
		}
		os << "\n";
	}
	return os;
}


////////////////////////////////////////////////////////////////////////////////
// Functions that provide sets of training data

// Generate 16 points of training data in the pattern illustrated in the project description
void GetTestData(std::vector<MVector>& x, std::vector<MVector>& y)
{
	x = { {0.125,.175}, {0.375,0.3125}, {0.05,0.675}, {0.3,0.025}, {0.15,0.3}, {0.25,0.5}, {0.2,0.95}, {0.15, 0.85},
		 {0.75, 0.5}, {0.95, 0.075}, {0.4875, 0.2}, {0.725,0.25}, {0.9,0.875}, {0.5,0.8}, {0.25,0.75}, {0.5,0.5} };

	y = { {1},{1},{1},{1},{1},{1},{1},{1},
		 {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1} };
}

// Generate 1000 points of test data in a checkerboard pattern
void GetCheckerboardData(std::vector<MVector>& x, std::vector<MVector>& y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	for (int i = 0; i < 1000; i++)
	{
		x[i] = { lr() / static_cast<double>(lr.max()),lr() / static_cast<double>(lr.max()) };
		double r = sin(x[i][0] * 12.5) * sin(x[i][1] * 12.5);
		y[i][0] = (r > 0) ? 1 : -1;
	}
}


// Generate 1000 points of test data in a spiral pattern
void GetSpiralData(std::vector<MVector>& x, std::vector<MVector>& y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	double twopi = 8.0 * atan(1.0);
	for (int i = 0; i < 1000; i++)
	{
		x[i] = { lr() / static_cast<double>(lr.max()),lr() / static_cast<double>(lr.max()) };
		double xv = x[i][0] - 0.5, yv = x[i][1] - 0.5;
		double ang = atan2(yv, xv) + twopi;
		double rad = sqrt(xv * xv + yv * yv);

		double r = fmod(ang + rad * 20, twopi);
		y[i][0] = (r < 0.5 * twopi) ? 1 : -1;
	}
}

// Save the the training data in x and y to a new file, with the filename given by "filename"
// Returns true if the file was saved succesfully
bool ExportTrainingData(const std::vector<MVector>& x, const std::vector<MVector>& y,
	std::string filename)
{
	// Check that the training vectors are the same size
	assert(x.size() == y.size());

	// Open a file with the specified name.
	std::ofstream f(filename);

	// Return false, indicating failure, if file did not open
	if (!f)
	{
		return false;
	}

	// Loop over each training datum
	for (unsigned i = 0; i < x.size(); i++)
	{
		// Check that the output for this point is a scalar
		assert(y[i].size() == 1);

		// Output components of x[i]
		for (int j = 0; j < x[i].size(); j++)
		{
			f << x[i][j] << " ";
		}

		// Output only component of y[i]
		f << y[i][0] << " " << std::endl;
	}
	f.close();

	if (f) return true;
	else return false;
}




////////////////////////////////////////////////////////////////////////////////
// Neural network class

class Network
{
public:

	// Constructor: sets up vectors of MVectors and MMatrices for
	// weights, biases, weighted inputs, activations and errors
	// The parameter nneurons_ is a vector defining the number of neurons at each layer.
	// For example:
	//   Network({2,1}) has two input neurons, no hidden layers, one output neuron
	//
	//   Network({2,3,3,1}) has two input neurons, two hidden layers of three neurons
	//                      each, and one output neuron
	Network(std::vector<unsigned> nneurons_)
	{
		nneurons = nneurons_;
		nLayers = nneurons.size();
		weights = std::vector<MMatrix>(nLayers);
		biases = std::vector<MVector>(nLayers);
		errors = std::vector<MVector>(nLayers);
		activations = std::vector<MVector>(nLayers);
		inputs = std::vector<MVector>(nLayers);
		// Create activations vector for input layer 0
		activations[0] = MVector(nneurons[0]);

		// Other vectors initialised for second and subsequent layers
		for (unsigned i = 1; i < nLayers; i++)
		{
			weights[i] = MMatrix(nneurons[i], nneurons[i - 1]);
			biases[i] = MVector(nneurons[i]);
			inputs[i] = MVector(nneurons[i]);
			errors[i] = MVector(nneurons[i]);
			activations[i] = MVector(nneurons[i]);
		}

		// The correspondence between these member variables and
		// the LaTeX notation used in the project description is:
		//
		// C++                      LaTeX
		// -------------------------------------
		// inputs[l-1][j-1]      =  z_j^{[l]}
		// activations[l-1][j-1] =  a_j^{[l]}
		// weights[l-1](j-1,k-1) =  W_{jk}^{[l]}
		// biases[l-1][j-1]      =  b_j^{[l]}
		// errors[l-1][j-1]      =  \delta_j^{[l]}
		// nneurons[l-1]         =  n_l
		// nLayers               =  L
		//
		// Note that, since C++ vector indices run from 0 to N-1, all the indices in C++
		// code are one less than the indices used in the mathematics (which run from 1 to N)
	}

	// Return the number of input neurons
	unsigned NInputNeurons() const
	{
		return nneurons[0];
	}

	// Return the number of output neurons
	unsigned NOutputNeurons() const
	{
		return nneurons[nLayers - 1];
	}

	// Evaluate the network for an input x and return the activations of the output layer
	MVector Evaluate(const MVector& x)
	{
		// Call FeedForward(x) to evaluate the network for an input vector x
		FeedForward(x);

		// Return the activations of the output layer
		return activations[nLayers - 1];
	}


	// Implement the training algorithm outlined in section 1.3.3
	// This should be implemented by calling the appropriate private member functions, below
	bool Train(const std::vector<MVector> x, const std::vector<MVector> y,
		double initsd, double learningRate, double costThreshold, int maxIterations)
	{
		// Check that there are the same number of training data inputs as outputs
		assert(x.size() == y.size());


		// TODO: Step 2 - initialise the weights and biases with the standard deviation "initsd"
		InitialiseWeightsAndBiases(initsd);

		for (int iter = 1; iter <= maxIterations; iter++)
		{
			// Step 3: Choose a random training data point i in {0, 1, 2, ..., N}
			int i = rnd() % x.size();


			// Step 4 - run the feed-forward algorithm
			FeedForward(x[i]);

			// Step 5 - run the back-propagation algorithm
			BackPropagateError(y[i]);
			//std::cout << "Step 5 complete" << std::endl;

			// Step 6 - update the weights and biases using stochastic gradient
			//          with learning rate "learningRate"
			UpdateWeightsAndBiases(learningRate);

			// Every so often, perform step 7 and show an update on how the cost function has decreased
			// Here, "every so often" means once every 1000 iterations, and also at the last iteration
			if ((!(iter % 100000)) || iter == maxIterations)
			{
				// Step 7(a) - calculate the total cost
				double Totalcost = TotalCost(x, y);

				// display the iteration number and total cost to the screen
				std::cout << "Iteration number: " << iter << std::endl;
				std::cout << "Total cost: " << Totalcost << std::endl;

				// Step 7(b) - return from this method with a value of true,
				//             indicating success, if this cost is less than "costThreshold".

				if (Totalcost < costThreshold)
				{
					// display the iteration number and total cost to the screen
					//std::cout << "Iteration number: " << iter << std::endl;
					//std::cout << "Total cost: " << Totalcost << std::endl;

					std::cout << "Success!!" << std::endl;

					FinalIter = iter;

					return true;
				}
			}

		} // Step 8: go back to step 3, until we have taken "maxIterations" steps

		// Step 9: return "false", indicating that the training did not succeed.
		return false;
	}

	bool TrainCost(const std::vector<MVector> x, const std::vector<MVector> y,
		double initsd, double learningRate, double costThreshold, int maxIterations, std::string filename)
	{
		// Check that there are the same number of training data inputs as outputs
		assert(x.size() == y.size());


		// TODO: Step 2 - initialise the weights and biases with the standard deviation "initsd"
		InitialiseWeightsAndBiases(initsd);

		std::ofstream question2cost;
		question2cost.open(filename);

		for (int iter = 1; iter <= maxIterations; iter++)
		{
			// Step 3: Choose a random training data point i in {0, 1, 2, ..., N}
			int i = rnd() % x.size();


			// Step 4 - run the feed-forward algorithm
			FeedForward(x[i]);

			// Step 5 - run the back-propagation algorithm
			BackPropagateError(y[i]);
			//std::cout << "Step 5 complete" << std::endl;

			// Step 6 - update the weights and biases using stochastic gradient
			//          with learning rate "learningRate"
			UpdateWeightsAndBiases(learningRate);

			// Every so often, perform step 7 and show an update on how the cost function has decreased
			// Here, "every so often" means once every 1000 iterations, and also at the last iteration
			if ((!(iter % 100)) || iter == maxIterations)
			{
				// Step 7(a) - calculate the total cost
				double Totalcost = TotalCost(x, y);

				question2cost.width(15); question2cost << iter;
				question2cost.width(15); question2cost << Totalcost << std::endl;

				// display the iteration number and total cost to the screen
				//std::cout << "Iteration number: " << iter << std::endl;
				//std::cout << "Total cost: " << Totalcost << std::endl;

				// Step 7(b) - return from this method with a value of true,
				//             indicating success, if this cost is less than "costThreshold".

				if (Totalcost < costThreshold)
				{
					// display the iteration number and total cost to the screen
					std::cout << "Learning rate: " << learningRate << std::endl;
					std::cout << "Iteration number: " << iter << std::endl;
					std::cout << "Total cost: " << Totalcost << std::endl;
					std::cout << "Success!!" << std::endl;

					FinalIter = iter;
					question2cost.close();
					return true;
				}
			}

		} // Step 8: go back to step 3, until we have taken "maxIterations" steps

		question2cost.close();
		// Step 9: return "false", indicating that the training did not succeed.
		return false;
	}


	// For a neural network with two inputs x=(x1, x2) and one output y,
	// loop over (x1, x2) for a grid of points in [0, 1]x[0, 1]
	// and save the value of the network output y evaluated at these points
	// to a file. Returns true if the file was saved successfully.
	bool ExportOutput(std::string filename)
	{
		// Check that the network has the right number of inputs and outputs
		assert(NInputNeurons() == 2 && NOutputNeurons() == 1);

		// Open a file with the specified name.
		std::ofstream f(filename);

		// Return false, indicating failure, if file did not open
		if (!f)
		{
			return false;
		}

		// generate a matrix of 250x250 output data points
		for (int i = 0; i <= 250; i++)
		{
			for (int j = 0; j <= 250; j++)
			{
				MVector out = Evaluate({ i / 250.0, j / 250.0 });
				f << out[0] << " ";
			}
			f << std::endl;
		}
		f.close();

		if (f) return true;
		else return false;
	}


	static bool Test();

	int FinalIter;

private:
	// Return the activation function sigma
	double Sigma(double z)
	{
		// Return sigma(z), as defined in equation (1.4)
		return tanh(z);
	}

	MVector Sigma(MVector v)
	{
		for (int i = 1; i <= v.size(); i++)
		{
			// Loop over the elements of the vector and apply Sigma
			v[i - 1] = Sigma(v[i - 1]);
		}

		return v;
	}

	// Return the derivative of the activation function
	double SigmaPrime(double z)
	{
		// Return d/dz(sigma(z)) = d/dz(tanch(z)) = sech^2(z)
		return 1 / (cosh(z) * cosh(z));
	}

	MVector SigmaPrime(MVector v)
	{
		for (int i = 1; i <= v.size(); i++)
		{
			// Loop over the elements of the vector and apply SigmaPrime
			v[i - 1] = SigmaPrime(v[i - 1]);
		}

		return v;
	}

	// Loop over all weights and biases in the network and set each
	// term to a random number normally distributed with mean 0 and
	// standard deviation "initsd"

	void InitialiseWeightsAndBiases(double initsd)
	{
		// Make sure the standard deviation supplied is non-negative
		assert(initsd >= 0);

		// Set up a normal distribution with mean zero, standard deviation "initsd"
		// Calling "dist(rnd)" returns a random number drawn from this distribution 
		std::normal_distribution<> dist(0, initsd);

		// Loop over all components of all the weight matrices
		// and bias vectors at each relevant layer of the network.

		for (unsigned l = 2; l <= nLayers; l++)
		{
			for (unsigned j = 1; j <= nneurons[l - 1]; j++)
			{
				for (unsigned k = 1; k <= nneurons[l - 2]; k++)
				{
					weights[l - 1](j - 1, k - 1) = dist(rnd);
				}
			}
		}

		for (unsigned l = 2; l <= nLayers; l++)
		{
			for (unsigned j = 1; j <= nneurons[l - 1]; j++)
			{
				biases[l - 1][j - 1] = dist(rnd);
			}
		}
	}

	// Evaluate the feed-forward algorithm, setting weighted inputs and activations
	// at each layer, given an input vector x
	void FeedForward(const MVector& x)
	{
		// Check that the input vector has the same number of elements as the input layer
		assert(x.size() == nneurons[0]);

		// Implement the feed-forward algorithm, equations (1.7), (1.8)

		activations[0] = x;

		for (unsigned l = 2; l <= nLayers; l++)
		{
			inputs[l - 1] = weights[l - 1] * activations[l - 2] + biases[l - 1];
			activations[l - 1] = Sigma(inputs[l - 1]);
		}
	}

	// Evaluate the back-propagation algorithm, setting errors for each layer 
	void BackPropagateError(const MVector& y)
	{
		// Check that the output vector y has the same number of elements as the output layer
		assert(y.size() == nneurons[nLayers - 1]);

		// Implement the back-propagation algorithm, equations (1.22) and (1.24)
		int L = nLayers - 1;

		errors[L] = SigmaPrime(inputs[L]) * (activations[L] - y);

		for (int l = L; l >= 2; l--)
		{
			errors[l - 1] = SigmaPrime(inputs[l - 1]) * TransposeTimes(weights[l], errors[l]);
		}
	}


	// Apply one iteration of the stochastic gradient iteration with learning rate eta.
	void UpdateWeightsAndBiases(double eta)
	{
		// Check that the learning rate is positive
		assert(eta > 0);

		// Update the weights and biases according to the stochastic gradient
		// iteration, using equations (1.25) and (1.26) to evaluate
		// the components of grad C.

		for (unsigned l = 2; l <= nLayers; l++)
		{
			for (unsigned j = 1; j <= nneurons[l - 1]; j++)
			{
				for (unsigned k = 1; k <= nneurons[l - 2]; k++)
				{
					weights[l - 1](j - 1, k - 1) -= eta * errors[l - 1][j - 1] *
						activations[l - 2][k - 1];
				}
			}
		}

		for (unsigned l = 2; l <= nLayers; l++)
		{
			for (unsigned j = 1; j <= nneurons[l - 1]; j++)
			{
				biases[l - 1][j - 1] -= eta * errors[l - 1][j - 1];
			}
		}

	}


	// Return the cost function of the network with respect to a single desired output y
	// Note: call FeedForward(x) first to evaluate the network output for an input x,
	//       then call this method Cost(y) with the corresponding desired output y
	double Cost(const MVector& y)
	{
		// Check that y has the same number of elements as the network has outputs
		assert(y.size() == nneurons[nLayers - 1]);

		// Return the cost associated with this output

		double sum = 0;

		for (int i = 1; i <= y.size(); i++)
		{
			sum += 0.5 * (y[i - 1] - activations[nLayers - 1][i - 1]) *
				(y[i - 1] - activations[nLayers - 1][i - 1]);
		}

		return sum;
	}

	// Return the total cost C for a set of training data x and desired outputs y
	double TotalCost(const std::vector<MVector> x, const std::vector<MVector> y)
	{
		// Check that there are the same number of inputs as outputs
		assert(x.size() == y.size());

		// Implement the cost function, equation (1.9), using
		// the FeedForward(x) and Cost(y) methods

		double sum = 0;

		for (unsigned i = 1; i <= x.size(); i++)
		{
			FeedForward(x[i - 1]);
			sum += Cost(y[i - 1]);
		}

		return sum / x.size();
	}

	// Private member data

	std::vector<unsigned> nneurons;
	std::vector<MMatrix> weights;
	std::vector<MVector> biases, errors, activations, inputs;
	unsigned nLayers;

};



bool Network::Test()
{
	// This function is a static member function of the Network class:
	// it acts like a normal stand-alone function, but has access to private
	// members of the Network class. This is useful for testing, since we can
	// examine and change internal class data.
	//
	// This function should return true if all tests pass, or false otherwise

	// An example test of FeedForward
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;

		// Call function to be tested with x = (0.3, 0.4)
		n.FeedForward({ 0.3, 0.4 });

		// Display the output value calculated
		//std::cout << n.activations[1][0] << std::endl;

		// Correct value is = tanh(0.5 + (-0.3*0.3 + 0.2*0.4))
		//                    = 0.454216432682259...
		// Fail if error in answer is greater than 10^-10:
		if (std::abs(n.activations[1][0] - 0.454216432682259) > 1e-10)
		{
			return false;
		}

		// Check that the input vector has been correctly filled
		if (std::abs(n.inputs[1][0] - 0.49) > 1e-10)
		{
			return false;
		}
	}

	// TODO: for each part of the Network class that you implement,
	//       write some more tests here to run that code and verify that
	//       its output is as you expect.
	//       I recommend putting each test in an empty scope { ... }, as 
	//       in the example given above.

	// A test of the vector version of Sigma
	{
		MVector x{ 1,2,3,4 };
		Network n({ 2, 4, 4, 1 });

		MVector Sigmax = n.Sigma(x);
		// Manually work out tanh(1), tanh(2), tanh(3), tanh(4).
		MVector answers{ 0.7615941559, 0.9640275800, 0.9950547536, 0.9993292997 };

		for (int i = 1; i <= 4; i++)
		{
			if (std::abs(Sigmax[i - 1] - answers[i - 1]) > 1e-10)
			{
				return false;
			}
			//std::cout << std::abs(Sigmax[i - 1] - answers[i - 1]) << std::endl;
		}

	}

	// A test of the vecor version of SigmaPrime
	{
		MVector x{ 1,2,3,4 };
		Network n({ 2, 4, 4, 1 });

		MVector Sigmax = n.SigmaPrime(x);
		// Manually work out sech^2(1), sech^2(2), sech^2(3), sech^2(4).
		MVector answers{ 0.4199743416 , 0.0706508248, 0.0098660371, 0.0013409506 };

		for (int i = 1; i <= 4; i++)
		{
			if (std::abs(Sigmax[i - 1] - answers[i - 1]) > 1e-10)
			{
				return false;
			}
			//std::cout << std::abs(Sigmax[i - 1] - answers[i - 1]) << std::endl;
		}

	}

	// A test of InitialiseWeightsAndBiases
	// As mentioned, this was not able to be automatically checked so
	// so was checked by hand and then commented out once verified.
	{

		Network n({ 2, 1 });
		// Initialise random weights and biases
		n.InitialiseWeightsAndBiases(1);
		// Display these to check that they look right
		// std::cout << n.biases[1][0] << " " << n.weights[1](0, 0) << " " << n.weights[1](0, 1) << std::endl;
		// Initialise them again and since it is random they should be different
		n.InitialiseWeightsAndBiases(1);
		// Display again to check
		// std::cout << n.biases[1][0] << " " << n.weights[1](0, 0) << " " << n.weights[1](0, 1) << std::endl;
	}

	// An example test of Cost
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 2 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;
		n.biases[1][1] = 0.3;
		n.weights[1](1, 0) = 0.1;
		n.weights[1](1, 1) = 0.4;

		// Call function to be tested with x = (0.3, 0.4)
		n.FeedForward({ 0.3, 0.4 });

		// Display the output value calculated
		//std::cout << n.activations[1][0] << std::endl;
		//std::cout << n.activations[1][1] << std::endl;

		MVector y{ 1.0,0.8 };

		//std::cout << (y - n.activations[1])[0] << std::endl;
		//std::cout << (y - n.activations[1])[1] << std::endl;

		//std::cout << (y - n.activations[1])[0] * (y - n.activations[1])[0] << std::endl;
		//std::cout << (y - n.activations[1])[1] * (y - n.activations[1])[1] << std::endl;	

		//std::cout << n.Cost(y) << std::endl;

		// Correct value is = tanh(0.5 + (-0.3*0.3 + 0.2*0.4))
		//                    = 0.454216432682259...
		// Fail if error in answer is greater than 10^-10:
		if (std::abs(n.Cost(y) - 0.20872298889) > 1e-10)
		{
			return false;
		}
	}

	// A test of BackPropagateError
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;

		// Call function to be tested with x = (0.3, 0.4)
		n.FeedForward({ 0.3, 0.4 });

		//std::cout << n.activations[1][0] << std::endl;
		//std::cout << "z[l] " << n.inputs[1][0] << std::endl;

		n.BackPropagateError({ 1.0 });

		//std::cout.precision(12); std::cout << n.errors[1][0] << std::endl;
		//std::cout << "-0.433181558125" << std::endl;

		if (std::abs(n.errors[1][0] - -0.433181558125) > 1e-10)
		{
			std::cout << "Error is " << n.errors[1][0] - -0.433181558125 << std::endl;
			return false;
		}
	}


	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Main function and example use of the Network class

// Create, train and use a neural network to classify the data in
// figures 1.1 and 1.2 of the project description.
//
// You should make your own copies of this function and change the network parameters
// to solve the other problems outlined in the project description.
void ClassifyTestData()
{
	// Create a network with two input neurons, two hidden layers of three neurons, and one output neuron
	Network n({ 2, 3, 3, 1 });

	// Get some data to train the network
	std::vector<MVector> x, y;

	GetTestData(x, y);

	// Train network on training inputs x and outputs y
	// Numerical parameters are:
	//  initial weight and bias standard deviation = 0.1
	//  learning rate = 0.1
	//  cost threshold = 1e-4
	//  maximum number of iterations = 100000
	bool trainingSucceeded = n.Train(x, y, 0.1, 0.1, 1e-4, 100000);

	// If training failed, report this
	if (!trainingSucceeded)
	{
		std::cout << "Failed to converge to desired tolerance." << std::endl;
	}

	// Generate some output files for plotting
	ExportTrainingData(x, y, "test_points.txt");
	n.ExportOutput("test_contour.txt");
}


int Question1Classify()
{
	Network n({ 2, 3, 3, 1 });
	std::vector<MVector> x, y;

	GetTestData(x, y);

	bool trainingSucceeded = n.Train(x, y, 0.1, 0.1, 1e-4, 100000);

	if (!trainingSucceeded)
	{
		return 0;
	}

	return n.FinalIter;
}

void Question1(unsigned n = 100)
{
	int success = 0;
	int sum = 0;

	for (unsigned i = 1; i <= n; i++)
	{
		int Q1 = Question1Classify();

		if (Q1 != 0)
		{
			success++;
			sum += Q1;
		}

		if (i % 10 == 0) std::cout << i << std::endl;
	}

	std::cout << "Number of fails: " << n - success << std::endl;
	std::cout << "Success rate: " << success / static_cast<double>(n) << std::endl;
	if (success != 0)
	{
		std::cout << "Average number of iterations for convergence on successes: "
			<< sum / success << std::endl;
	}
}

int Question2Average(double eta, unsigned N, int& success)
{
	Network n({ 2, 3, 3, 1 });
	std::vector<MVector> x, y;
	GetTestData(x, y);

	int sum = 0;

	for (unsigned i = 1; i <= N; i++)
	{
		n.Train(x, y, 0.1, eta, 1e-4, 100000);

		int Q2 = n.FinalIter;

		if (Q2 != 0)
		{
			success++;
			sum += Q2;
		}
	}

	if (success != 0)
	{
		return sum / success;
	}

	return 0;
}

void Question2()
{
	std::ofstream question2;
	question2.open("question2.txt");

	for (unsigned eta = 400; eta <= 500 - 1; eta += 5)
	{
		if (eta == 6) eta = 5;
		double etasmall = eta * 0.001;
		std::cout << "Eta: " << etasmall << std::endl;

		int success = 0;
		int N = 200;

		int Q2 = Question2Average(etasmall, N, success);

		std::cout << "Average iterations: " << Q2 << std::endl;
		std::cout << "Successes: " << success << "/" << N << std::endl;
		std::cout << std::endl;

		question2.width(6); question2 << etasmall;
		if (Q2 == 0) { question2.width(10); question2 << 100000; }
		else { question2.width(10); question2 << Q2; }
		question2.width(10); question2 << static_cast<double>(success) / N *
			100 << std::endl;
	}

	for (unsigned eta = 4; eta <= 10; eta += 1)
	{
		double etasmall = eta * 0.1;
		std::cout << "Eta: " << etasmall << std::endl;

		int success = 0;
		int N = 1000;

		int Q2 = Question2Average(etasmall, N, success);

		std::cout << "Average iterations: " << Q2 << std::endl;
		std::cout << "Successes: " << success << "/" << N << std::endl;
		std::cout << std::endl;

		question2.width(6); question2 << etasmall;
		if (Q2 == 0) { question2.width(10); question2 << 100000; }
		else { question2.width(10); question2 << Q2; }
		question2.width(10); question2 << static_cast<double>(success) / N *
			100 << std::endl;
	}

	question2.close();
}

void Question2Cost()
{
	Network n({ 2, 3, 3, 1 });
	std::vector<MVector> x, y;
	GetTestData(x, y);

	n.TrainCost(x, y, 1.0, 0.001, 1e-4, 2200000, "question2Cost1.txt");
	n.TrainCost(x, y, 1.0, 0.002, 1e-4, 2200000, "question2Cost2.txt");
	n.TrainCost(x, y, 1.0, 0.005, 1e-4, 2200000, "question2Cost3.txt");
	n.TrainCost(x, y, 1.0, 0.01, 1e-4, 2200000, "question2Cost4.txt");
	n.TrainCost(x, y, 1.0, 0.05, 1e-4, 2200000, "question2Cost5.txt");
	n.TrainCost(x, y, 1.0, 0.1, 1e-4, 2200000, "question2Cost6.txt");
	n.TrainCost(x, y, 1.0, 0.2, 1e-4, 2200000, "question2Cost7.txt");
	n.TrainCost(x, y, 1.0, 0.3, 1e-4, 2200000, "question2Cost8.txt");
	n.TrainCost(x, y, 1.0, 1.0, 1e-4, 2200000, "question2Cost9.txt");
}

int Question3Average(double initsd, unsigned N, int& success)
{
	Network n({ 2, 3, 3, 1 });
	std::vector<MVector> x, y;
	GetTestData(x, y);

	int sum = 0;

	for (unsigned i = 1; i <= N; i++)
	{
		n.Train(x, y, initsd, 0.1, 1e-4, 100000);

		int Q3 = n.FinalIter;

		if (Q3 != 0)
		{
			success++;
			sum += Q3;
		}
	}

	if (success != 0)
	{
		return sum / success;
	}

	return 0;
}

void Question3()
{
	std::ofstream question3;
	question3.open("question3.txt");

	unsigned MAX = 0;

	for (unsigned initsd = 1, increment = 1, counter = 1; initsd <= MAX; initsd += increment)
	{
		double initsdsmall = initsd * 0.001;
		std::cout << "Initial standard deviation: " << initsdsmall << std::endl;

		int success = 0;
		int N = 200;

		int Q3 = Question3Average(initsdsmall, N, success);
		std::cout << "Average iterations: " << Q3 << std::endl;
		std::cout << "Successes: " << success << "/" << N << std::endl;
		std::cout << std::endl;

		question3.width(6); question3 << initsdsmall;
		if (Q3 == 0) { question3.width(10); question3 << 100000; }
		else { question3.width(10); question3 << Q3; }
		question3.width(10); question3 << static_cast<double>(success) / N * 100 << std::endl;

		if ((initsd % (increment * 10)) == 0) increment *= 10;

		++counter;
	}

	MAX = 100;

	for (unsigned initsd = 1, increment = 1; initsd <= MAX; initsd += increment)
	{
		double initsdsmall = initsd * 1.0;
		std::cout << "Initial standard deviation: " << initsdsmall << std::endl;

		int success = 0;
		int N = 1;

		int Q3 = Question3Average(initsdsmall, N, success);
		std::cout << "Average iterations: " << Q3 << std::endl;
		std::cout << "Successes: " << success << "/" << N << std::endl;
		std::cout << std::endl;

		question3.width(6); question3 << initsdsmall;
		if (Q3 == 0) { question3.width(10); question3 << 100000; }
		else { question3.width(10); question3 << Q3; }
		question3.width(10); question3 << static_cast<double>(success) / N * 100 << std::endl;

		if ((initsd % (increment * 10)) == 0) increment *= 10;

	}

	question3.close();
}

void Question4()
{
	// Hidden layers: 0

	{
		Network n({ 2, 1 });
		std::vector<MVector> x, y;
		GetCheckerboardData(x, y);

		std::cout << "Network { 2, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.3, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 20,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_0pointsspiral.txt");
		n.ExportOutput("question4_0contourspiral.txt");
	}


	// Hidden layers: 1

	{
		Network n({ 2, 20, 1 });
		std::vector<MVector> x, y;
		GetCheckerboardData(x, y);

		std::cout << "Network { 2, 20, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.3, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 10,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_1pointsspiral.txt");
		n.ExportOutput("question4_1contourspiral.txt");
	}


	// Hidden layers: 2

	{
		Network n({ 2, 20, 20, 1 });
		std::vector<MVector> x, y;
		GetCheckerboardData(x, y);

		std::cout << "Network { 2, 20, 20, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.01, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 20,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_2points.txt");
		n.ExportOutput("question4_2contour.txt");
	}


	// Hidden layers: 3

	{
		Network n({ 2, 20, 20, 20, 1 });
		std::vector<MVector> x, y;
		GetCheckerboardData(x, y);

		std::cout << "Network { 2, 20, 20, 20, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.001, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 10,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_3points.txt");
		n.ExportOutput("question4_3contour.txt");
	}

	// Spiral data

	// No hidden layers
	{
		Network n({ 2, 1 });
		std::vector<MVector> x, y;
		GetSpiralData(x, y);

		std::cout << "Network { 2, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.3, 10000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 20,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_0pointsspiral.txt");
		n.ExportOutput("question4_0contourspiral.txt");
	}


	// Hidden layers: 1

	{
		Network n({ 2, 20, 1 });
		std::vector<MVector> x, y;
		GetSpiralData(x, y);

		std::cout << "Network { 2, 20, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.3, 10000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 10,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_1pointsspiral.txt");
		n.ExportOutput("question4_1contourspiral.txt");
	}


	// Hidden layers: 2

	{
		Network n({ 2, 20, 20, 1 });
		std::vector<MVector> x, y;
		GetSpiralData(x, y);

		std::cout << "Network { 2, 20, 20, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.01, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 20,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_2pointsspiral.txt");
		n.ExportOutput("question4_2contourspiral.txt");
	}


	// Hidden layers: 3

	{
		Network n({ 2, 20, 20, 20, 1 });
		std::vector<MVector> x, y;
		GetSpiralData(x, y);

		std::cout << "Network { 2, 20, 20, 20, 1 }" << std::endl;

		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.01, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 10,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "question4_3pointsspiral.txt");
		n.ExportOutput("question4_3contourspiral.txt");
	}

}

void GenerateChequerboardPlot()
{
	{
		Network n({ 2, 20, 20, 1 });
		std::vector<MVector> x, y;
		GetCheckerboardData(x, y);

		std::cout << "Network { 2, 20, 20, 1 }" << std::endl;

		// Initial standard deviation, learning rate, cost threshold, maximum iterations
		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.01, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 20,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "ChequerboardPlotPoints.txt");
		n.ExportOutput("ChequerboardPlotContour.txt");
	}
}

void GenerateSpiralPlot()
{
	{
		Network n({ 2, 20, 20, 1 });
		std::vector<MVector> x, y;
		GetSpiralData(x, y);

		std::cout << "Network { 2, 20, 20, 1 }" << std::endl;
		// Training data x, y, initial standard deviation, learning rate, cost threshold, maximum iterations.
		bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.01, 20000000);

		// If training failed, report this
		if (!trainingSucceeded)
		{
			std::cout << "Failed to converge to desired tolerance in 20,000,000 iterations." << std::endl;
		}

		ExportTrainingData(x, y, "SpiralPlotPoints.txt");
		n.ExportOutput("SpiralPlotContour.txt");
	}
}

int main()
{
	// Call the test function
	bool testsPassed = Network::Test();

	// If tests did not pass, something is wrong; end program now
	if (!testsPassed)
	{
		return 1;
	}

	// Tests passed, so run our program.
	std::cout << "Tests passed, so run our program." << std::endl;

	GenerateChequerboardPlot();
	
	GenerateSpiralPlot();

	return 0;
}