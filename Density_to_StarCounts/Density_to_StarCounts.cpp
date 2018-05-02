#include <iostream>
#include <math.h>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <limits>

#include "tao/asynchronous_algorithms/particle_swarm.hxx"
#include "tao/asynchronous_algorithms/differential_evolution.hxx"

#include "tao/synchronous_algorithms/parameter_sweep.hxx"
#include "tao/synchronous_algorithms/synchronous_gradient_descent.hxx"
#include "tao/synchronous_algorithms/synchronous_newton_method.hxx"

//from undvc_common
#include "undvc_common/arguments.hxx"
#include "undvc_common/vector_io.hxx"
using namespace std;

// Discrete_Convolution performs the discrete convolution on inputs "list1" and then places the 
// returns the result in a vector of doubles.
//
// This function assumes the following: 
// list1 are initialized with values at each array index
// we are always convolving with a gaussian centered at 4 with a standard deviation of .6
vector<double> Discrete_Convolution(vector<double> list1)
{
	vector<double> gaussian;
	vector<double> histogram = list1;
	
	//a gaussian with standard deviation of .6
	//centered at 4, gaussians are found by integrating 
	//the gaussian from the bin size which are shown below
	//for each bin
	
	//bin 1 (2.75 - 3.25)
	gaussian.push_back(.0870393);
	//bin 2 (3.25 - 3.75)
	gaussian.push_back(.232811);
	//bin 3  (3.75 - 4.25)
	gaussian.push_back(.32078);
	//bin 4 (4.25 - 4.75)
	gaussian.push_back(.232811);
	//bin 5 (4.75 - 5.25)
	gaussian.push_back(.0870393);

	vector<double> result; 
	vector<double> included;
	
	//runs a convolution. As we need to make sure we always convolve with a histogram of 
	//size 1, we create another array that is maps a histogram bar to another a histogram of 
	//equal length. This histogram holds 1's or 0's and will depend on whether its corresponding 
	//guassian bar is on the boundary or not. 
	for (unsigned int t = 0; t < gaussian.size(); t++)
		included.push_back(1);

	for (unsigned int i = 0; i < histogram.size(); i++)
	{
		for (unsigned int t = 0; t < gaussian.size(); t++)
			included[t] = 1;
		for (int j = -2; j <= 2; j++)
		{
			if ((i+j < 0) || (i+j >= histogram.size()))
				included[j+2] = 0; 
		}	
		double overall = 0.;
		for (unsigned int alpha = 0; alpha < gaussian.size(); alpha++)
		{
			overall += included[alpha] * gaussian[alpha]; 
		}
		double total = 0.;
		for (unsigned int k = 0; k < gaussian.size(); k++)
		{
			if (included[k] == 0)
				continue; 
			total += histogram[i+k-2] * gaussian[k] / overall; 
		}
		result.push_back(total);
	}
	return result; 
}


//returns distance in units of kPc given a double representing the apparent 
//magnitude. This function is used for turnoff stars with an absolute magnitude 
//of 4 
double find_distance(double apparent_mag)
{
	double absolute_mag = 4;
	double exponent = ((apparent_mag - absolute_mag)/5);
	double result = pow(10.0, exponent) / 100.; 
	return result; 
}

//given a lower bound magnitude and an upper bound magnitude and the interval
//this function returns a vector holding the volumes from the min magnitude to the
//max magnitude with the interval size in units of kPc^3
vector<double> Generate_Volumes(double Mag_min, double Mag_max, double interval)
{
	int counter = 0;
	double apparent_mag2, distance1, distance2; 
	double Omega = 2.5 * 2.5; //solid angle (units stradians) 
	
	double size = (Mag_max - Mag_min) / interval;
	int array_size = (int)size;
	
	vector<double> volume_list (array_size, 0);
	
	apparent_mag2 = Mag_min + counter * interval;
	distance2 = find_distance(apparent_mag2);
	
	//uses equation of solid angle to find volume 
	for (counter = 0; counter < array_size; counter++)
	{
		distance1 = distance2; 
		apparent_mag2 = Mag_min + (counter+1) * interval; 
		distance2 = find_distance(apparent_mag2); 
		volume_list[counter] = Omega/3 *((pow(distance2, 3) - pow(distance1, 3)));
	}
	return volume_list;
}

//runs the completeness coefficient equation to return a vector of doubles 
//corresponding to each respective magnitude's completeness coefficient
vector<double> Completeness(double Mag_min, double Mag_max, double interval)
{
	double s0 = .9402;
	double s1 = 1.6171;
	double s2 = 23.5877;
	double iterations = (Mag_max - Mag_min) / interval; 
	vector<double> CC; 
	for (double i = 0; i < iterations; i++)
	{
		double value = s0 / (exp(s1*(i * .5 + Mag_min + .25 - s2)) + 1);
		CC.push_back(value);
	}
	return CC; 
}

//if array1.size() == array2.size()
//  then returns a vector defined as vector[i] = array1[i] * array2[i] for all i
//if array1.size() != array2.size()
//  then return a vector of size array1 with vector[i] = 0 for all i
vector<double> mult2arrays(const vector<double> &array1, const vector<double> &array2)
{
	int size1 = array1.size();
	int size2 = array2.size();
	vector<double> result (size1, 0);
	//checks the two array sizes are the same
	if (size1 != size2)
	{
		return result;
	}
	//preforms the multiplication
	for (int i = 0; i < size1; i++)
	{
		result[i] = array1[i] * array2[i]; 
	}
	
	return result; 
}

//if observed.size != expected.size return -1
//else returns the chi-square test of fit
double chi_squared(vector<double> observed, vector<double> expected)
{
	if (observed.size() != expected.size())
		return -1;
	double chi_sq = 0; 
	//finds the chi-square and adds it to the overall sum 
	for (unsigned int i = 0; i < observed.size(); i++)
	{
		chi_sq += (observed[i] - expected[i])*(observed[i] - expected[i])/expected[i];
	}
	return chi_sq; 
}


//function that takes input (2 histograms with lengths) with output (chi-squared value)
//t1 is the guess of density of stars, t2 is the expected star counts
//Temporarily add global variable since TAO can only handle objective functions with 1 argument
//Storage is a global variable that will hold the final best fit result once the optimization is done
vector<double> objective_function_t2;
vector<double> storage; 

double objective_function(const vector<double> &t1)
{
	storage = t1;
	//lowerbound and upperbound magnitudes
	double lowerbound = 16;
	double upperbound = 16 + double(t1.size())/2.0;
	
	vector<double> starcounts = t1;

	
	//convolves with the gaussian
	vector<double> convolved = Discrete_Convolution(starcounts);
	
	//finds the completeness coefficients
	vector<double> CC = Completeness(lowerbound, upperbound, .5);

	//because the convolution adds extra bins on the end, this removes the bins
	//to make the two vectors have the same size
	//this is just a safegaurd
	while (convolved.size() != CC.size())
	{
		convolved.pop_back();
		convolved.erase(convolved.begin());
	}
	
	//multiplies by the completeness
	vector<double> final_star_count = mult2arrays(convolved, CC);
	
	//finds the chi-square value 
	double result = chi_squared(final_star_count, objective_function_t2);
	
	//returns the negative result as the optimizer seeks maximize, but chi-square want
	//the smallest value 
	return -(result);
}

//takes a single histogram, runs the gradient descent
void optimize(vector<double> t1)
{
	//some needed vectors 
        std::vector<double> min_bound(t1.size(), 0.0);
        std::vector<double> max_bound(t1.size(), 100);
        vector<double> step_size(t1.size(), 1);
        vector<double> starting_point = t1;
	double lowerbound = 16;
	double upperbound = 16+double(t1.size())/2.;
        vector<double> completeness = Completeness(lowerbound, upperbound, .5);
        for(unsigned int i = 0; i < t1.size(); i++)
        {
               starting_point[i] = starting_point[i] / completeness[i];
        }
	vector<double> final_parameters;
	double final_fitness = 0;
	vector <string> args;
	args.push_back("--min_improvement");
	
	//min improvement
	args.push_back("1e-06");
        synchronous_gradient_descent(args, objective_function, starting_point, step_size, final_parameters, final_fitness);
}

//takes a made up star count and transforms it to a simulated data
vector<double> transformation(vector<double> t1)
{
	double lowerbound = 16;
	double upperbound = 16 + double(t1.size())/2.0;
	
	vector<double> starcounts = t1;
	for (unsigned int i = 0; i < starcounts.size(); i++)
	{
		if (i == starcounts.size() - 1)
			cout<<starcounts[i]<<endl;
		else
			cout<<starcounts[i]<<", ";
	}


	//convolves with the gaussian
	cout<<"above convolved with the gaussian: ";
	vector<double> convolved = Discrete_Convolution(starcounts);
	for (unsigned int i = 0; i < convolved.size(); i++)
	{
		if (i == convolved.size() - 1)
			cout<<convolved[i]<<endl;
		else
			cout<<convolved[i]<<", ";
	}

	
	vector<double> CC = Completeness(lowerbound, upperbound, .5);
	while (convolved.size() != CC.size())
	{
		convolved.pop_back();
		convolved.erase(convolved.begin());
	}
	
	cout<<"above multiplied by the completeness coefficient ";
	vector<double> final_star_count = mult2arrays(convolved, CC);
	for (unsigned int i = 0; i < final_star_count.size(); i++)
	{
		if (i == final_star_count.size() - 1)
			cout<<final_star_count[i]<<endl;
		else
			cout<<final_star_count[i]<<", ";
	}


	return final_star_count;
}

int main()
{
	//representing made up data
	vector<double> t1; 
	t1.push_back(9);
	t1.push_back(8);
	t1.push_back(7);	
	t1.push_back(6);
	t1.push_back(5);
	t1.push_back(4);
	t1.push_back(3);
	t1.push_back(2);
	t1.push_back(1);
	
	//creates a vector which has same size as t1 and is what will be fit
	vector<double> t2; 
	for (unsigned int i = 0; i < t1.size(); i++)
	{
		t2.push_back(5);
	}	

	std::cout<<"transformed histogram"<<std::endl;
	
	
	//transforms t1 to simulated data
	vector<double> data = transformation(t1);
	
	//printing for debugging
	for (unsigned int i = 0; i < data.size(); i++)
	{
		std::cout<<t1[i]<<" transformed to: "<<data[i]<<endl;
	}	
	
	//sets objective functoin and optimizes 
	objective_function_t2 = data;
	optimize(t2);
	
	//printing for debugging
	std::cout<<"Starting numbers "<<endl;
	for (unsigned int i = 0; i < t1.size(); i++)
	{
		if (i == t1.size()-1)
			cout<<t1[i]<<endl;
		else 
			cout<<t1[i]<<", ";	
	}
	cout<<"Starting numbers transformed: \n";
	data = transformation(t1);	
	cout<<"\n";
	
	//prints the best fit storage 
	cout<<"Storage: \n";
	for (unsigned int w = 0; w < storage.size(); w++)
	{
		if (w == storage.size()-1)
			cout<<storage[w]<<endl;
		else
			cout<<storage[w]<<", ";
	}
	cout<<"Storage transformed: \n";
	vector<double> temp = transformation(storage);
}
