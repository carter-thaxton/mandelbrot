//! Saved places, recalled with the number keys.
//!
//! Slot 0 is home and can't be overwritten; slots 1-9 are the user's, and
//! start out filled with [`PRESETS`]. Kept in memory for now. A [`Place`]
//! holds everything needed to persist them later: its center prints and
//! parses as exact decimal strings ([`crate::bigfixed::Fixed::to_decimal`] /
//! [`Place::parse`]), and its span doesn't depend on the window size.

use crate::app::Place;

/// The reserved home slot.
pub const HOME: u8 = 0;
/// User slots are 1..=USER_SLOTS.
pub const USER_SLOTS: u8 = 9;

/// A built-in place: exact decimal center, view height, and a short name.
pub struct Preset {
    pub name: &'static str,
    pub x: &'static str,
    pub y: &'static str,
    pub span: f64,
}

/// Starting bookmarks, shallowest first, each showing a different kind of
/// structure. The minibrot centers are nuclei located with
/// `examples/find_nucleus.rs` (Newton's method in `Fixed` precision), exact
/// to well below each view's pixel size:
///
/// - period 204, size 2.9e-16, in the elephant valley (slots 2-3)
/// - period 17, size 2.0e-19, on the needle near -2 (slot 4)
/// - period 114, size 1.2e-82, next to the Misiurewicz point c = i (7-9)
///
/// The seahorse coordinate (1, 5) is the well-known 33-digit one, which
/// sits beside a period-8007 minibrot of size 9e-33; c = i (6) is exact.
pub const PRESETS: [Preset; USER_SLOTS as usize] = [
    Preset {
        name: "seahorse valley",
        x: "-0.743643887037158704752191506114774",
        y: "0.131825904205311970493132056385139",
        span: 2.6e-8,
    },
    Preset {
        name: "elephant crossroads",
        x: "0.286931865508042059229267",
        y: "0.014286693150767741452436",
        span: 3e-13,
    },
    Preset {
        name: "elephant valley minibrot",
        x: "0.286931865508042059229267",
        y: "0.014286693150767741452436",
        span: 1.2e-15,
    },
    Preset { name: "needle minibrot", x: "-1.999999999138270118607796920", y: "0", span: 8e-19 },
    Preset {
        name: "spiral sea",
        x: "-0.743643887037158704752191506114774",
        y: "0.131825904205311970493132056385139",
        span: 2.6e-20,
    },
    Preset { name: "dendrite at c = i", x: "0", y: "1", span: 1e-50 },
    Preset {
        name: "six-armed star",
        x: "0.000000000000000000000000000000000000000073794373395757068405341127766922571943656582466811",
        y: "1.000000000000000000000000000000000000000031025811934975204283677154524596606390686207486972",
        span: 1e-61,
    },
    Preset {
        name: "spiral galaxy",
        x: "0.000000000000000000000000000000000000000073794373395757068405341127766922571943656582466811",
        y: "1.000000000000000000000000000000000000000031025811934975204283677154524596606390686207486972",
        span: 1e-71,
    },
    Preset {
        name: "minibrot at 1e-82",
        x: "0.000000000000000000000000000000000000000073794373395757068405341127766922571943656582466811",
        y: "1.000000000000000000000000000000000000000031025811934975204283677154524596606390686207486972",
        span: 5e-82,
    },
];

/// A saved place, with a name if it came from a preset.
#[derive(Clone, Debug, PartialEq)]
struct Bookmark {
    place: Place,
    name: Option<String>,
}

#[derive(Default)]
pub struct Bookmarks {
    slots: [Option<Bookmark>; USER_SLOTS as usize],
}

impl Bookmarks {
    /// Slots 1-9 filled with [`PRESETS`].
    pub fn with_presets() -> Self {
        let mut b = Self::default();
        for (slot, p) in (1..).zip(&PRESETS) {
            let place = Place::parse(p.x, p.y, p.span).expect("preset coordinates parse");
            b.slots[slot - 1] = Some(Bookmark { place, name: Some(p.name.to_string()) });
        }
        b
    }

    /// Save `place` in user slot 1-9, replacing what was there (name
    /// included). Returns false (and saves nothing) for the home slot or
    /// anything out of range.
    pub fn save(&mut self, slot: u8, place: Place) -> bool {
        match Self::index(slot) {
            Some(i) => {
                self.slots[i] = Some(Bookmark { place, name: None });
                true
            }
            None => false,
        }
    }

    /// The place saved in user slot 1-9, if any. Home is not stored here (it
    /// depends on nothing saved): see `Explorer::home_place`.
    pub fn get(&self, slot: u8) -> Option<&Place> {
        self.slots[Self::index(slot)?].as_ref().map(|b| &b.place)
    }

    /// The name of the place in slot 1-9, if it has one.
    pub fn name(&self, slot: u8) -> Option<&str> {
        self.slots[Self::index(slot)?].as_ref()?.name.as_deref()
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

    #[test]
    fn presets_fill_every_user_slot_shallow_to_deep() {
        let b = Bookmarks::with_presets();
        let mut names = std::collections::HashSet::new();
        let mut last_span = f64::INFINITY;
        for slot in 1..=USER_SLOTS {
            let p = b.get(slot).unwrap_or_else(|| panic!("slot {slot} empty"));
            assert!(p.span < 1e-7, "slot {slot} is shallow: {}", p.span);
            assert!(p.span < last_span, "slot {slot} not deeper than the one before");
            last_span = p.span;
            assert!(names.insert(b.name(slot).unwrap()), "duplicate name in slot {slot}");
        }
        assert!(b.get(HOME).is_none());
    }

    #[test]
    fn saving_over_a_preset_drops_its_name() {
        let mut b = Bookmarks::with_presets();
        assert_eq!(b.name(4), Some("needle minibrot"));
        b.save(4, place(0.1, 1e-5));
        assert_eq!(b.name(4), None);
    }
}
