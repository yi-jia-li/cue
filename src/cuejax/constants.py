import astropy.constants as c
import astropy.units as u

c_cms = c.c.cgs.value
c_AAs = c.c.to(u.AA / u.s).value
c_kms = c.c.to(u.km / u.s).value

Lsun_cgs = (1*u.Lsun).cgs.value
lyman_limit = 912. # AA