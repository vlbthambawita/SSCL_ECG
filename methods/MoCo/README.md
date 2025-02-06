1. Network Architecture

MoCo uses two neural networks:
	•	Query Encoder  f_q : Encodes the “query” image into a feature vector.
	•	Key Encoder  f_k : Encodes the “key” images into feature vectors. This is a momentum-based moving average of the query encoder  f_q .

Momentum Update for  f_k :

The parameters of the key encoder ( \theta_k ) are updated using a moving average of the query encoder ( \theta_q ):

\theta_k \gets m \cdot \theta_k + (1 - m) \cdot \theta_q

Where:
	•	 m  is the momentum coefficient (e.g., 0.999 or 0.99).

This update stabilizes training by ensuring  f_k  changes more slowly than  f_q .