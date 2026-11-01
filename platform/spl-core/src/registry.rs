use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{DynOp, Op, OpDescriptor};

/// Every op a host can run. Built once at startup from the app catalog.
#[derive(Default, Clone)]
pub struct Registry {
    ops: BTreeMap<&'static str, Arc<dyn DynOp>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Panics on a duplicate ID: two apps claiming one ID is a build bug.
    pub fn op<T: Op>(&mut self, op: T) -> &mut Self {
        let previous = self.ops.insert(T::ID, Arc::new(op));
        assert!(previous.is_none(), "op `{}` registered twice", T::ID);
        self
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn DynOp>> {
        self.ops.get(id).cloned()
    }

    pub fn describe(&self) -> Vec<OpDescriptor> {
        self.ops.values().map(|op| op.descriptor()).collect()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}
