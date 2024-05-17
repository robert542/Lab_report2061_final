import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def model(n, B):
    f1 = y_data[0] 
    return f1 * n * np.sqrt(1 + B * n**2)

x_data = np.array([1, 3, 5, 7, 9, 11])
y_data = np.array([110.15, 334.4, 581.1, 825.0, 1079.72, 1348.65])

#fit
params, params_covariance = curve_fit(lambda n, B: model(n, B), x_data, y_data, p0=[0.01])
fitted_y = model(x_data, *params)

#residual
residuals = y_data - fitted_y
# Calculate total sum of squares (SST)
sst = np.sum((y_data - np.mean(y_data))**2)
#sum residual squared
ssr = np.sum(residuals**2)
# Calculate R^2
r_squared = 1 - (ssr / sst)


print("Fitted B value:", params[0])
print("Residuals:", residuals)
print("R^2:", r_squared)

#graph of fit against data
plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, fitted_y, label='Fitted function', color='red')
plt.xlabel('Harmonic (n)')
plt.ylabel('frequency (HZ)')
plt.title('Inharmonicity Fit to Collected Data')
plt.legend()
plt.show()
#plot of residual
plt.figure(figsize=(6, 4))
plt.stem(x_data, residuals)
plt.xlabel('n')
plt.ylabel('Residuals (HZ)')
plt.title('Residual Plot of Fit')
plt.show()
