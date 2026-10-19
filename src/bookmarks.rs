//! Saved places, recalled with the number keys.
//!
//! Slot 0 is home and can't be overwritten; slots 1-9 are the user's. Kept in
//! memory for now. A [`Place`] holds everything needed to persist them later:
//! its center prints and parses as exact decimal strings
//! ([`crate::bigfixed::Fixed::to_decimal`] / [`crate::bigfixed::Fixed::parse`]),
//! and its span doesn't depend on the window size.

use crate::app::Place;

/// The reserved home slot.
pub const HOME: u8 = 0;
/// User slots are 1..=USER_SLOTS.
pub const USER_SLOTS: u8 = 9;

#[derive(Default)]
pub struct Bookmarks {
    slots: [Option<Place>; USER_SLOTS as usize],
}

impl Bookmarks {
    /// Save `place` in user slot 1-9, replacing what was there. Returns false
    /// (and saves nothing) for the home slot or anything out of range.
    pub fn save(&mut self, slot: u8, place: Place) -> bool {
        match Self::index(slot) {
            Some(i) => {
                self.slots[i] = Some(place);
                true
            }
            None => false,
        }
    }

    /// The place saved in user slot 1-9, if any. Home is not stored here (it
    /// depends on nothing saved): see `Explorer::home_place`.
    pub fn get(&self, slot: u8) -> Option<&Place> {
        self.slots[Self::index(slot)?].as_ref()
    }

    fn index(slot: u8) -> Option<usize> {
        (1..=USER_SLOTS).contains(&slot).then(|| slot as usize - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigfixed::Fixed;

    fn place(x: f64, span: f64) -> Place {
        Place { center: (Fixed::from_f64(x, 2), Fixed::zero(2)), span }
    }

    #[test]
    fn save_and_recall_user_slots() {
        let mut b = Bookmarks::default();
        assert!(b.get(3).is_none());
        assert!(b.save(3, place(0.25, 1e-10)));
        assert_eq!(b.get(3), Some(&place(0.25, 1e-10)));
        // Overwrite.
        assert!(b.save(3, place(-0.5, 1e-20)));
        assert_eq!(b.get(3), Some(&place(-0.5, 1e-20)));
        // Slots are independent.
        assert!(b.save(9, place(0.1, 1.0)));
        assert_eq!(b.get(3).unwrap().span, 1e-20);
    }

    #[test]
    fn home_and_out_of_range_slots_are_not_writable() {
        let mut b = Bookmarks::default();
        assert!(!b.save(HOME, place(0.25, 1e-10)));
        assert!(!b.save(10, place(0.25, 1e-10)));
        assert!(b.get(HOME).is_none());
        assert!(b.get(10).is_none());
    }
}
