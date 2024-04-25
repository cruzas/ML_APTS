# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from datetime import datetime, timedelta


# # Work Package labels
# wp_labels = ['WP1: Dataset (UD,UI)', 'WP2: Development of NN (UD)', 
#              'WP3: Post-processing (UI)', 'WP4: 3D Model (UD,UI)']
# wp_start_dates = [datetime(2025, 1, 1), datetime(2025, 3, 1), datetime(2025, 6, 1), datetime(2026, 3, 1)]
# wp_end_dates = [datetime(2025, 6, 1), datetime(2026, 3, 1), datetime(2026, 6, 1), datetime(2026, 12, 1)]


# # Adjusting the approach to correctly handle datetime objects for rectangle plotting
# fig, ax = plt.subplots(figsize=(10, 6))
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# # Adding vertical lines for each rectangle start and end
# for i, (start, end, label, color) in enumerate(zip(wp_start_dates, wp_end_dates, wp_labels, colors)):
#     # Calculating the width as the number of days
#     width = (end - start).days
#     # Converting width from days into matplotlib's date format
#     width_in_date_format = mdates.date2num(end) - mdates.date2num(start)
#     # Creating a rectangle
#     rect = plt.Rectangle((mdates.date2num(start), i - 1), width_in_date_format, 1, color=color)
#     ax.add_patch(rect)
#     # Adding WP number inside the rectangle
#     # ax.text(mdates.date2num(start) + width_in_date_format/2, i-0.5, label.split(': ')[0], 
#     #         va='center', ha='center', color='white', weight='bold')
#     ax.text(mdates.date2num(start) + width_in_date_format/2, i-0.5, label, 
#             va='center', ha='center', color='black', weight='bold')
#     ax.axvline(x=start, ymin=0, ymax=i/4, color='gray', linestyle='--', zorder=-1)
#     ax.axvline(x=end, ymin=0, ymax=i/4, color='gray', linestyle='--', zorder=-1)
    
    
# # Adjusting y-axis to not show labels
# plt.yticks(range(len(wp_labels)), ['']*len(wp_labels))
# # Beautify the plot
# plt.title('Project Workflow Timeline')
# ax.set_ylim(-1, len(wp_labels)-1)
# start_date = min(wp_start_dates)
# ax.set_xlim(start_date - timedelta(days=30), wp_end_dates[-1] + timedelta(days=30))
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', linewidth=0.5)
# plt.tight_layout()

# plt.show()





from sympy import symbols, Eq, solve, And

# Define the symbols
x, y, z = symbols('x y z')

# Define the equation
equation = Eq((x*90 + y*60)*1.05 + z, 400000)

# Define the constraints
constraints = [
    z >= 10000, 
    z <= 20000,
    y >= 2500,
    y <= 3500,
    x >= 800,
    x <= 1500
]

# Solve the system of equations including the equation and constraints
# Note: Sympy's solve function may have difficulty finding solutions for complex systems of inequalities directly.
# Alternatively, you might explore numerical solutions or use optimization packages for complex constraints.
solution = solve([equation] + constraints, (x, y, z))

solution
