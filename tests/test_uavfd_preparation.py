import unittest
import numpy as np

from prepare_uavfd import FS, LENGTH, identity, noise_replay_metadata, select_nonoverlapping_test, zoh


def row(index, start, flight='test_flight', segment=0):
    return dict(window_id=index, split='eval', flight_id=flight, segment=segment,
                start_boot_s=start/FS, end_boot_s=(start+LENGTH-1)/FS)


class UAVFDPreparationTests(unittest.TestCase):
    def test_selection_starts_at_first_valid_window_not_modulo1024(self):
        rows=[row(0,512),row(1,1024),row(2,1536)]
        selected,chains=select_nonoverlapping_test(rows)
        self.assertEqual(selected,[0,2])
        self.assertEqual(chains[0]['selected'],2)

    def test_selection_restarts_after_rejected_window_gap(self):
        rows=[row(0,0),row(1,512),row(2,2048),row(3,2560),row(4,3072)]
        selected,chains=select_nonoverlapping_test(rows)
        self.assertEqual(selected,[0,2,4])
        self.assertEqual([r['original'] for r in chains],[2,3])

    def test_segment_boundaries_are_not_joined(self):
        rows=[row(0,0,segment=0),row(1,512,segment=0),
              row(2,2048,segment=1),row(3,2560,segment=1)]
        selected,chains=select_nonoverlapping_test(rows)
        self.assertEqual(selected,[0,2]);self.assertEqual(len(chains),2)

    def test_zero_order_hold_does_not_use_future_or_stale_values(self):
        values,valid=zoh(np.array([1.,2.,3.]),np.array([[10.],[20.],[30.]]),
                         np.array([.9,1.,1.04,1.2,2.,3.1]),.05)
        np.testing.assert_array_equal(valid,[False,True,True,False,True,False])
        np.testing.assert_array_equal(values[:,0],[10.,10.,10.,10.,20.,30.])

    def test_filename_class_and_group_mapping(self):
        self.assertEqual(identity('NO_FAULT6.mat'),(6,0,0))
        self.assertEqual(identity('FAULT_M5_5.mat'),(5,5,1))
        self.assertEqual(identity('FAULT_M1_10.mat'),(1,10,2))

    def test_original_test_noise_positions_preserve_the_old_bank_subset(self):
        rows=[row(7,0),row(8,512),row(9,1024)]
        selected,_=select_nonoverlapping_test(rows)
        meta=noise_replay_metadata(rows,selected)
        self.assertEqual(meta['test_noise_original_count'],3)
        self.assertEqual(meta['test_noise_positions'],[0,2])
        full=np.random.RandomState(meta['test_noise_seed']).standard_normal((3,9,1024)).astype(np.float32)
        replay=full[meta['test_noise_positions']]
        np.testing.assert_array_equal(replay[0],full[0])
        np.testing.assert_array_equal(replay[1],full[2])
        fresh=np.random.RandomState(meta['test_noise_seed']).standard_normal((2,9,1024)).astype(np.float32)
        self.assertFalse(np.array_equal(replay[1],fresh[1]))


if __name__=='__main__':unittest.main()
