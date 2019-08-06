/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Latest edit: Tobias Hascher
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <time.h>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */  
  std::default_random_engine gen;
  normal_distribution<double> x_norm_dist(x, std[0]);
  normal_distribution<double> y_norm_dist(y, std[1]);
  normal_distribution<double> theta_norm_dist(theta, std[2]);
  
  num_particles = 200;  // TODO: Set the number of particles
  Particle part;
  // Loop through all particles and initialize them
  for (int i=0; i<num_particles; ++i) {
    part.id = i;
    part.x = x_norm_dist(gen);
    part.y = y_norm_dist(gen);
    part.theta = theta_norm_dist(gen);
    part.weight = 1.0;
    // add it to particles vector
    particles.push_back(part);
    weights.push_back(part.weight);
  }
  is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> x_norm_dist(0.0, std_pos[0]);
  normal_distribution<double> y_norm_dist(0.0, std_pos[1]);
  normal_distribution<double> theta_norm_dist(0.0, std_pos[2]);
  for (int i=0; i<num_particles; ++i) {
    if (fabs(yaw_rate) > 0.00001) {
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta))  +  x_norm_dist(gen);
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)))  +  y_norm_dist(gen);
      particles[i].theta += yaw_rate*delta_t  +  theta_norm_dist(gen);
    }
    else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) +  x_norm_dist(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) +  y_norm_dist(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // Loop through all Lidar observations
  for (unsigned int i=0; i<observations.size(); ++i) {
    double min_dist = 10000.0;
    double this_dist = 10000.0;
    // Loop through all map landmarks
    for (unsigned int j=0; j<predicted.size(); ++j) {
      this_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (this_dist < min_dist) {
        min_dist = this_dist;
        observations[i].id = j;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, 
                                   double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  //std::cout<<"Update weights ..."<<std::endl;
  double weights_sum = 0.0;
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  // Calculate the normalization term
  double norm_term = 1/(2*M_PI*sig_x*sig_y);
  
  // Loop through all particles
  for (int i=0; i<num_particles; ++i) {
    vector<LandmarkObs> observations_glob;
    // transform oberservations into map coordinates
    for (unsigned int j=0; j<observations.size(); ++j) {
      LandmarkObs lm_temp;
      lm_temp.x = particles[i].x + (observations[j].x * cos(particles[i].theta) - (observations[j].y * sin(particles[i].theta)));
      lm_temp.y = particles[i].y + (observations[j].x * sin(particles[i].theta) + (observations[j].y * cos(particles[i].theta)));
      lm_temp.id = 0;
      observations_glob.push_back(lm_temp);
    }
    // map landmarks within Sensor range
    vector<LandmarkObs> predicted;
    LandmarkObs pred_temp;
    for (unsigned int k=0; k<map_landmarks.landmark_list.size(); ++k) {
      double dist_to_part = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
      if (dist_to_part <= sensor_range) {
        pred_temp.x = map_landmarks.landmark_list[k].x_f;
        pred_temp.y = map_landmarks.landmark_list[k].y_f;
        pred_temp.id = map_landmarks.landmark_list[k].id_i;
        predicted.push_back(pred_temp);
      }
    }
    // Data Association:  associates observations_glob.id with the referring map landmark id
    dataAssociation(predicted, observations_glob);
    
    double multiv_gauss_obs = 1;
    double particle_weight = 1;
    // Loop through all obervations
    for (unsigned int l=0; l<observations_glob.size(); ++l) {
      int lm_id = observations_glob[l].id;
      double x_obs = observations_glob[l].x;
      double ux = predicted[lm_id].x;
      double y_obs = observations_glob[l].y;
      double uy = predicted[lm_id].y;
      double exp_1, exp_2;
      // Calculate Exponent Parts
      if ((sig_x != 0) && (sig_y != 0)) {
        exp_1 = pow((x_obs - ux),2) / (2*pow(sig_x,2));
        exp_2 = pow((y_obs - uy),2) / (2*pow(sig_y,2));
      }
      else {
        std::cout<<"sig_x, sig_y div/0 error in update weights method"<<std::endl;
      }
      // for this observation: Calculate the multi-variate Gaussian probability density function
      multiv_gauss_obs = norm_term * exp(-(exp_1+exp_2));
      // multiply it up
      //if (i==0) //tobi
        //std::cout<<multiv_gauss_obs<<" "; // tobi, print all multiv_gauss_obs of 0th particle
      if (multiv_gauss_obs < 0.000001)
        multiv_gauss_obs = 0.000001;
      particle_weight *= multiv_gauss_obs; // product of all multivariate Gaussians
    }
    particles[i].weight = particle_weight;
    //std::cout<<particle_weight<<" "; // tobi print all particle weights
    weights_sum += particle_weight;
  }
  std::cout<<std::endl; //tobi
  double weights_sum_norm = 0.0;
  // Finally normalize the particles weights
  for(int m=0; m<num_particles; ++m) {
    particles[m].weight /= weights_sum;
    weights_sum_norm += particles[m].weight;
    weights[m] = particles[m].weight;
  }
  std::cout<<"Update_weights: Weights sum = "<<weights_sum<<"   Weights sum normalized = "<<weights_sum_norm<<std::endl;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // resampling wheel //
  vector<Particle> resampled_particles;
  std::default_random_engine gen, re;
  std::uniform_int_distribution<int> uni_dist(0, num_particles-1);
  std::uniform_real_distribution<double> random_double(0.0, 1.0);
  
  int index = uni_dist(gen); // generate random particle index (between 0 and 999)
  double beta = 0.0;
  double max_weight = *max_element(weights.begin(), weights.end());
  double weight_sum = 0.0; //tobi
  
  for (int i=0; i<num_particles; ++i) {
    beta += random_double(re) * 2 * max_weight;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index+1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
    weights[i] = particles[index].weight;
    weight_sum += particles[index].weight;//tobi
  }
  std::cout<<"Resample: max weight: "<<max_weight<<"   Weights sum = "<<weight_sum<<std::endl;;
  //Substitution of the particles
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}