from numba.experimental import jitclass
from numba import float64

spec = [
  ('x', float64[:]),
  ('y', float64[:]),
  ('vx', float64[:]),
  ('vy', float64[:]),
  ('ax', float64[:]),
  ('ay', float64[:]),
  ('pot', float64[:]),
]

@jitclass(spec)
class State(object):
  def __init__(self, x, y, vx, vy, ax, ay, pot):
    self.x = x
    self.y = y
    self.vx = vx
    self.vy = vy
    self.ax = ax
    self.ay = ay
    self.pot = pot
