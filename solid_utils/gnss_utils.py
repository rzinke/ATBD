import numpy as np
import math
import warnings


class Site:
    """A location at which a measurement is taken, e.g., by GNSS OR INSAR,
    and the measurement quantities collected there.

    All units should be in meters and years (for velocities).
    Derived quantities (e.g., velocities, steps) should include uncertainties.

    One should be able to easily compute the distance between two sites,
    and the difference in derived quantity (velocity; step) with
    uncertainty propagation.
    """
    def __init__(self, site:str, site_lon:float, site_lat:float):
        """Every site will have a name and location.
        """
        # Ascribe parameters
        self.site = site
        self.site_lon = site_lon
        self.site_lat = site_lat

    def write_velocity(self, vel:float, vel_err:float):
        """Write the values of a velocity measurement to the object.
        Units should be m, yr.
        """
        # Parse velocity estimates
        self.vel = vel
        self.vel_err = vel_err

    def report(self):
        print_str = f"{self.site:s}"
        if hasattr(self, 'vel'):
            print_str += f" {1000*self.vel:.2f} +- {1000*self.vel_err:.3f} mm/yr"

        return print_str

    def __sub__(self, other):
        """Subtract the velocity of one site from the current one,
        and propagate the uncertainty.
        The returned object inherits the name and position of the
        original object.
        """
        res_site = Site(self.site, self.site_lon, self.site_lat)

        # Difference in velocities
        vel = self.vel - other.vel
        vel_err = (self.vel_err**2 + other.vel_err**2) ** 0.5
        res_site.write_velocity(vel, vel_err)

        return res_site
