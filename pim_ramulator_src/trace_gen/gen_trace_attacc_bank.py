import argparse
import math
import copy
import numpy as np

# Page-wise attention configuration
PAGE_SIZE = 16  # tokens per page
PAGE_SELECT_RATIO = 0.25  # use 1/4 of pages for each new token
TOKENS_PER_ROW = PAGE_SIZE  # each page (16 tokens) fits in one row
KV_BUDGET          = 0  

# Page management
def calculate_page_config(L):
    """Calculate page configuration for given sequence length"""
    total_pages = math.ceil(L / PAGE_SIZE)
    active_pages = max(1, math.ceil(total_pages * PAGE_SELECT_RATIO))
    return total_pages, active_pages

# === (2) 替换 page 数计算逻辑 ===
def calculate_page_config_fixed_budget(L, kv_budget):
    """
    固定 KV_Budget 版本：
      - PAGE_SIZE 行容量 (pair/row) 已知
      - ROW_BUDGET = ceil(kv_budget / PAGE_SIZE)
      - active_pages = min(ROW_BUDGET, total_pages)
    """
    total_pages  = math.ceil(L / PAGE_SIZE)
    row_budget   = math.ceil(kv_budget / PAGE_SIZE)
    active_pages = min(row_budget, total_pages)
    return total_pages, active_pages


def print_page_statistics(L, total_pages, active_pages, verbose=False):
    """Print detailed page statistics"""
    effective_tokens = active_pages * PAGE_SIZE
    unused_pages = total_pages - active_pages
    memory_reduction = (1 - effective_tokens / L) * 100
    
    print(f" Page Statistics:")
    print(f"   ├─ Sequence: {L:,} tokens")
    print(f"   ├─ Page size: {PAGE_SIZE} tokens/page")
    print(f"   ├─ Total pages: {total_pages}")
    print(f"   ├─ Active pages: {active_pages} ({(active_pages/total_pages)*100:.1f}%)")
    print(f"   ├─ Unused pages: {unused_pages} ({(unused_pages/total_pages)*100:.1f}%)")
    print(f"   ├─ Effective tokens: {effective_tokens:,}")
    print(f"   └─ Memory reduction: {memory_reduction:.1f}%")
    
    if verbose:
        print(f" Page Layout:")
        for i in range(min(10, total_pages)):  # Show first 10 pages
            start_token = i * PAGE_SIZE
            end_token = min(start_token + PAGE_SIZE - 1, L - 1)
            status = " ACTIVE" if i >= (total_pages - active_pages) else " UNUSED"
            print(f"   Page {i:2d}: tokens {start_token:4d}-{end_token:4d} {status}")
        
        if total_pages > 10:
            print(f"   ... and {total_pages - 10} more pages")
            # Show last few pages if they're active
            if active_pages > 0:
                last_active_start = total_pages - active_pages
                if last_active_start >= 10:
                    print(f"   Last active pages:")
                    for i in range(max(last_active_start, total_pages - 3), total_pages):
                        start_token = i * PAGE_SIZE
                        end_token = min(start_token + PAGE_SIZE - 1, L - 1)
                        print(f"   Page {i:2d}: tokens {start_token:4d}-{end_token:4d}  ACTIVE")

def get_active_page_addresses(base_addr, total_pages, active_pages, page_id):
    """Get addresses of active pages for current token"""
    # For simplicity, use the most recent pages (can be changed to other strategies)
    active_page_indices = list(range(max(0, total_pages - active_pages), total_pages))
    
    page_addresses = []
    for page_idx in active_page_indices:
        # Each page starts at a row boundary
        page_addr = base_addr + page_idx * HBM_GS['row']
        page_addresses.append(page_addr)
    
    return page_addresses, active_page_indices

# Original global variables
model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16 # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # byte
n_mac = 16

 
# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row'] 
HBM_GS['bg']      = n_bank * HBM_GS['ba'] 
HBM_GS['rank']     = n_bg * HBM_GS['bg'] 
HBM_GS['pch']     = n_rank * HBM_GS['rank'] 
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm'] 


## --------------------------------------  HBM memory space -----------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1   | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

# Our implementation 
cmd_norm_mvgb  = []
cmd_norm_mac  = []
cmd_norm_mvsb = []
cmd_cos_mvgb  = []
cmd_cos_mac  = []
cmd_cos_mvsb = []

valid_channels = []

# def cmd_list_reset():
#   cmd_score_wrgb   = []
#   cmd_score_mac    = []
#   cmd_score_mvsb   = []
#   cmd_sfm          = []
#   cmd_context_mvgb = []
#   cmd_context_mac  = []
#   cmd_context_mvsb = []

#   valid_channel = []

def cmd_list_reset():
    global cmd_score_wrgb, cmd_score_mac, cmd_score_mvsb, cmd_sfm
    global cmd_context_mvgb, cmd_context_mac, cmd_context_mvsb, valid_channels

    cmd_score_wrgb.clear()
    cmd_score_mac.clear()
    cmd_score_mvsb.clear()
    cmd_sfm.clear()
    cmd_context_mvgb.clear()
    cmd_context_mac.clear()
    cmd_context_mvsb.clear()
    valid_channels.clear()
    cmd_norm_mvgb.clear()
    cmd_norm_mac.clear()
    cmd_norm_mvsb.clear()
    cmd_cos_mvgb.clear()
    cmd_cos_mac.clear()
    cmd_cos_mvsb.clear()

"""
Notice: 
  From my point of view, it seems that, the only thing need to be edited is the "L"
  If you set "L" to effective_L, you need to edit nothing else (except clustering overhead).
  You current code add unreasonable amount of MV_SB command compared to orginal AttAcc code.
  To make the trace command number reasonable, 
  I disabled all edition you made over original code in this function,
  except the change of "L" to "effective_L"
"""

def Attention(L, key_addr, val_addr, itr, valid_channel = n_channel, 
              page_wise=False, iter_num = 0, cluster_width=64, norm_addr=None, 
              cluater_itr=0, add_cluster=False):
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])
  cmd_sfm.append([])
  cmd_context_mvgb.append([])
  cmd_context_mac.append([])
  cmd_context_mvsb.append([])

  cmd_norm_mvgb.append([])
  cmd_norm_mac.append([])
  cmd_norm_mvsb.append([])
  cmd_cos_mvgb.append([])
  cmd_cos_mac.append([])
  cmd_cos_mvsb.append([])

  valid_channels.append(valid_channel);
  
  # --- 计算 page 配置 ---
  if page_wise:
      if KV_BUDGET > 0:  # 固定 budget 路径
          total_pages, active_pages = calculate_page_config_fixed_budget(L, KV_BUDGET)
      else:              # 沿用原比例逻辑
          total_pages, active_pages = calculate_page_config(L)
      effective_L = active_pages * PAGE_SIZE      # 真正参与计算的 token 数
  else:
      effective_L = L


  def cluster_norm_squre(addr_offset, L, cluster_width=64):
    
    # In cluster all K will do sum(K**2) regardless of sparsity.
    for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):# 16 
      cmd_norm_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / n_bank / n_mac)): # 2
        idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_norm_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
          ## parallelization

      # MVSB command (Move to sqrt buffer next to host) 
      if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
        cmd_norm_mvsb[itr].append([])
        for bg_idx in range(n_bg):   
          for rank in range(n_rank):
            for lch in range(math.ceil(valid_channel)):
              bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                          bg_idx * HBM_GS['bg']
              hex_addr = hex(bank_addr)[2:]
              cmd_norm_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))
              # Sqrt write back

     
  def cluater_norm_divide(addr_offset, L, cluster_width=64):
    # Get sqrt back from sqrt engine
    for rank in range(n_rank):
      for bg_idx in range(n_bg):
        for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
          # number of columns of partition = L / (R parallel units)
            for lch in range(math.ceil(valid_channel)):
              # GEMV buffer address, col granularity = 1
              addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                  bg_idx * HBM_GS['bg'] + col_idx
              hex_addr = hex(addr)[2:]
              cmd_norm_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    # Compute K / |K|, use data and multiplier at same position for all L vectors
    for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):# 16 
      cmd_norm_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / n_bank / n_mac)): # 2
        idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_norm_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))


  def cluster_similar(addr_offset, L, cluster_width=64):
    """
      Multiply all K vectors with all centroids, 
      Number of centroids = L/cluster_width
      This only happens for 16 times for per 64 tokens.
      Each time, we can start at same K address, we do not need new memory space for this.
    """
    for clu_idx in range(cluater_itr):
      # Get centoid from buffer
      for n_idx in range(math.ceil(L*L/cluster_width / n_pch / n_rank / n_bg)):
        cmd_cos_mac[itr].append([])
        # Do computation of cosine similarity.
        for k_idx in range(math.ceil(dhead / n_bank / n_mac)): 
          idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

          # All bank command (legacy channel)
          for lch in range(math.ceil(valid_channel)):
            addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
            hex_addr = hex(addr)[2:]
            cmd_cos_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
  
        # MVSB command (Move similarity to host) 
        if n_idx % 16 == 15 or n_idx == math.ceil(L*L/cluster_width / n_pch / n_rank / n_bg) - 1:
          cmd_cos_mvsb[itr].append([])
          for bg_idx in range(n_bg):   
            for rank in range(n_rank):
              for lch in range(math.ceil(valid_channel)):
                bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                          bg_idx * HBM_GS['bg']
                hex_addr = hex(bank_addr)[2:]
                cmd_cos_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))
                

  def cluster_update(addr_offset, L, cluster_width=64):
    for clu_idx in range(cluater_itr):
      # Write back sorted vector group. each vector will have a group idx. the length of all idx is L
      for rank in range(n_rank):
        for bg_idx in range(n_bg):
          for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
            # number of columns of partition = L / (R parallel units)
            for lch in range(math.ceil(valid_channel)):
              # GEMV buffer address, col granularity = 1
              addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                  bg_idx * HBM_GS['bg'] + col_idx
              hex_addr = hex(addr)[2:]
              cmd_cos_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))
      """          
        Compute new cetroid and update: We directly igonore this overhead, 
        since in cluater_norm_divide(addr_offset, L) the divide should be single MUL, but we calculate MAC.
        The average and update process has very similar overhead compare to additional ADD in cluater_norm_divide(addr_offset, L) 
      """ 

    # Write new centroids
    for clu_idx in range(cluater_itr):
      # Write back ranked similarity
      # Only pch* MAC is enough for this parallel
      for col_idx in range(math.ceil(L/cluster_width/ (n_pch * n_mac))):
        # number of columns of partition = L / (R parallel units)
        for lch in range(math.ceil(valid_channel)):
          # GEMV buffer address, col granularity = 1
          addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
              bg_idx * HBM_GS['bg'] + col_idx
          hex_addr = hex(addr)[2:]
          cmd_cos_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))


  def cluster_remap(addr_offset, L):
    # Reorder all KV.
    # Data broadcasting for pch, rank, bg, and ba
    for ba_idx in range(n_bank): # number of partitions
      for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
        for lch in range(math.ceil(valid_channel)):
          # GEMV buffer address, col granularity = 1
          addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
          hex_addr = hex(addr)[2:]
          cmd_cos_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))     

  """
  if not debug:

    def score_cpvec(addr_offset, L):
      ## (pCH) C, C, R, R (MAC)
      ## write input vector to gemv buffer
      # number of partition = (R parallel units)

      # Data broadcasting for pch, rank, bg, and ba
        for ba_idx in range(n_bank): # number of partitions
          for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
            for lch in range(math.ceil(valid_channel)):
              # GEMV buffer address, col granularity = 1
              addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
              hex_addr = hex(addr)[2:]
              cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def score_mac(addr_offset, L):
      ## (pCH) C, C, R, R (MAC)
      # MAC and move output vector to softmax buffer
      ## Vector (1 x k) x Matrix (k x n) multiplication
      ## GEMV unit = adder tree mode
    
      # Use effective_L for page-wise computation
      computation_L = effective_L if page_wise else L
    
      # For page-wise attention, generate addresses for active pages only
      if page_wise:
        page_addresses, active_page_indices = get_active_page_addresses(
            addr_offset, total_pages, active_pages, itr)
      
        # Process each active page

        for page_idx, page_addr in enumerate(page_addresses):
          for n_idx in range(math.ceil(PAGE_SIZE / n_pch / n_rank / n_bg)):
            cmd_score_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
              idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
            
              # All bank command targeting specific page (row)
              for lch in range(math.ceil(valid_channel)):
                # Address calculation for page-wise access
                addr = page_addr + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                hex_addr = hex(addr)[2:]
                cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
          
            # MVSB command for this page
            if n_idx % 16 == 15 or n_idx == math.ceil(PAGE_SIZE / n_pch / n_rank / n_bg) - 1:
              cmd_score_mvsb[itr].append([])
              for bg_idx in range(n_bg):   
                for rank in range(n_rank):
                  for lch in range(math.ceil(valid_channel)):
                    bank_addr = page_addr + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                              bg_idx * HBM_GS['bg']
                    hex_addr = hex(bank_addr)[2:]
                    cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))
      else:
        # Original implementation for non-page-wise
        for n_idx in range(math.ceil(computation_L / n_pch / n_rank / n_bg)):
          cmd_score_mac[itr].append([])
          for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
            idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

            # All bank command (legacy channel)
            for lch in range(math.ceil(valid_channel)):
              addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
              hex_addr = hex(addr)[2:]
              cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

          ## MVSB command (Move to Softmax buffer) 
          if n_idx % 16 == 15 or n_idx == math.ceil(computation_L / n_pch / n_rank / n_bg) - 1:
            cmd_score_mvsb[itr].append([])
            for bg_idx in range(n_bg):   
              for rank in range(n_rank):
                for lch in range(math.ceil(valid_channel)):
                  bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                            bg_idx * HBM_GS['bg']
                  hex_addr = hex(bank_addr)[2:]
                  cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))
  
    ## (pCH) R, R, C, C (MAC)
    def context_cpvec(addr_offset, L):
      ## write input vector to gemv buffer
      ## number of partition = (BG and BA banks)

      # Use effective_L for page-wise computation
      computation_L = effective_L if page_wise else L

      # For page-wise attention, process active pages only
      if page_wise:
        page_addresses, active_page_indices = get_active_page_addresses(
            addr_offset, total_pages, active_pages, itr)
      
        # Process each active page
        for page_idx, page_addr in enumerate(page_addresses):
          for rank in range(n_rank):
            for bg_idx in range(n_bg):
              for col_idx in range(math.ceil(PAGE_SIZE / (n_pch * n_rank * n_bg * n_mac))):
                for lch in range(math.ceil(valid_channel)):
                  # GEMV buffer address, col granularity = 1
                  addr = page_addr + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                       bg_idx * HBM_GS['bg'] + col_idx
                  hex_addr = hex(addr)[2:]
                  cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))
      else:
        # Original implementation for non-page-wise
        # Data broadcasting for bg and ba
        for rank in range(n_rank):
          for bg_idx in range(n_bg):
            for col_idx in range(math.ceil(computation_L / (n_pch * n_rank * n_bg * n_mac))):
              # number of columns of partition = L / (R parallel units)
                for lch in range(math.ceil(valid_channel)):
                  # GEMV buffer address, col granularity = 1
                  addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                       bg_idx * HBM_GS['bg'] + col_idx
                  hex_addr = hex(addr)[2:]
                  cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    def context_mac(addr_offset, L):
      # MAC and move output vector to softmax buffer
      ## Vector (1xk) x Matrix (k x n ) multiplication
      ## GEMV unit = mac mode
    
      # Use effective_L for page-wise computation
      computation_L = effective_L if page_wise else L
    
      # For page-wise attention, process active pages only
      if page_wise:
        page_addresses, active_page_indices = get_active_page_addresses(
            addr_offset, total_pages, active_pages, itr)
      
        # Process each active page for context computation
        for page_idx, page_addr in enumerate(page_addresses):
          for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
            cmd_context_mac[itr].append([])
            for k_idx in range(math.ceil(PAGE_SIZE / (n_pch * n_rank * n_bg))):
              idx = k_idx + n_idx * math.ceil(PAGE_SIZE / (n_pch * n_rank * n_bg))
              for lch in range(math.ceil(valid_channel)):
                addr = page_addr + lch * HBM_GS['ch'] + idx * HBM_GS['col'] 
                hex_addr = hex(addr)[2:]
                cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            # Generate MVSB commands for this page
            cmd_context_mvsb[itr].append([])
            for ba_idx in range(n_bank):
              for rank in range(n_rank):
                for lch in range(math.ceil(valid_channel)):
                  bank_addr = page_addr + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                            ba_idx * HBM_GS['ba'] 
                  hex_addr = hex(bank_addr)[2:]
                  cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))
      else:
        # Original implementation for non-page-wise
        for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
          cmd_context_mac[itr].append([])
          for k_idx in range(math.ceil(computation_L / (n_pch * n_rank * n_bg))):
            idx = k_idx + n_idx * math.ceil(computation_L / (n_pch * n_rank * n_bg))
            for lch in range(math.ceil(valid_channel)):
              addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col'] 
              hex_addr = hex(addr)[2:]
              cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

          ## parallelization. Generate 16 elements per n_idx
          cmd_context_mvsb[itr].append([])
          for ba_idx in range(n_bank):
            for rank in range(n_rank):
              for lch in range(math.ceil(valid_channel)):
                bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                          ba_idx * HBM_GS['ba'] 
                hex_addr = hex(bank_addr)[2:]
                cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def softmax(L):
      for lch in range(math.ceil(valid_channel)):
        addr = lch * HBM_GS['ch'] 
        hex_addr = hex(addr)[2:]
        cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))
  """

  def score_cpvec(addr_offset, L):
    ## (pCH) C, C, R, R (MAC)
    ## write input vector to gemv buffer
    # number of partition = (R parallel units)

    # Data broadcasting for pch, rank, bg, and ba
    for ba_idx in range(n_bank): # number of partitions
      for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
        for lch in range(math.ceil(valid_channel)):
          # GEMV buffer address, col granularity = 1
          addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
          hex_addr = hex(addr)[2:]
          cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))
        
  def score_mac(addr_offset, L):
    ## (pCH) C, C, R, R (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1 x k) x Matrix (k x n) multiplication
    ## GEMV unit = adder tree mode
    for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):# 16 
      cmd_score_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / n_bank / n_mac)): # 2
        idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
          ## parallelization

      # MVSB command (Move to Softmax buffer) 
      ## A output element is generated for every n_idx
      if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
        cmd_score_mvsb[itr].append([])
        for bg_idx in range(n_bg):   
          for rank in range(n_rank):
            for lch in range(math.ceil(valid_channel)):
              bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                        bg_idx * HBM_GS['bg']
              hex_addr = hex(bank_addr)[2:]
              cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    ## (pCH) R, R, C, C (MAC)
  def context_cpvec(addr_offset, L):
    ## write input vector to gemv buffer
    ## number of partition = (BG and BA banks)

    # Data broadcasting for bg and ba
    for rank in range(n_rank):
      for bg_idx in range(n_bg):
        for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
          # number of columns of partition = L / (R parallel units)
            for lch in range(math.ceil(valid_channel)):
              # GEMV buffer address, col granularity = 1
              addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                    bg_idx * HBM_GS['bg'] + col_idx
              hex_addr = hex(addr)[2:]
              cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

  def context_mac(addr_offset, L):
    # MAC and move output vector to softmax buffer
    ## Vector (1xk) x Matrix (k x n ) multiplication
    ## GEMV unit = mac mode
    for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
      cmd_context_mac[itr].append([])
      for k_idx in range(math.ceil(L / (n_pch * n_rank * n_bg))):
        idx = k_idx + n_idx * math.ceil(L / (n_pch * n_rank * n_bg))
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col'] 
          hex_addr = hex(addr)[2:]
          cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

      ## parallelization. Generate 16 elements per n_idx
      cmd_context_mvsb[itr].append([])
      for ba_idx in range(n_bank):
        for rank in range(n_rank):
          for lch in range(math.ceil(valid_channel)):
            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                        ba_idx * HBM_GS['ba'] 
            hex_addr = hex(bank_addr)[2:]
            cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def softmax(L):
    for lch in range(math.ceil(valid_channel)):
      addr = lch * HBM_GS['ch'] 
      hex_addr = hex(addr)[2:]
      cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))
  
  if False:
    cluster_norm_squre(norm_addr, L, cluster_width)
    cluater_norm_divide(norm_addr, L, cluster_width)
    cluster_similar(norm_addr, L, cluster_width)
    cluster_update(norm_addr, L, cluster_width)
    cluster_remap(norm_addr, L)

  score_cpvec(key_addr, effective_L)

  score_mac(key_addr, effective_L)

  #print(cmd_cluster_mvsb)
  softmax(effective_L)

  context_cpvec(val_addr, effective_L)

  context_mac(val_addr, effective_L)
  """
  num = 0
  for i in range(len(cmd_score_mac)):
    for j in range(len(cmd_score_mac[i])):
       for n in range(len(cmd_score_mac[i][j])):
          num+=len(cmd_score_mac[i][j][n])
  print(num)
  num = 0
  for i in range(len(cmd_context_mac)):
    for j in range(len(cmd_context_mac[i])):
       for n in range(len(cmd_context_mac[i][j])):
          num+=len(cmd_context_mac[i][j][n])
  print(num)
  num = 0
  for i in range(len(cmd_score_mvsb)):
    for j in range(len(cmd_score_mvsb[i])):
       for n in range(len(cmd_score_mvsb[i][j])):
          num+=len(cmd_score_mvsb[i][j][n])
  print(num)
  
  #exit()
      #print(len(cmd_score_mvsb[i][j]))
  #print('='*20)
  
  #for i in range(len(cmd_context_mvsb)):
  #  print(len(cmd_context_mvsb[i]))
    #for j in range(len(cmd_context_mvsb[i])):
      #print(len(cmd_context_mvsb[i][j]))
  #print('='*20)
  """


"""
 n_head and n_req = n_req per a HBM 
 def run_attention(dhead, n_head_per_hbm, L, trace_file_name, page_wise=False):
   partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
   head_offset = partition_size
   v_offset = pow(2, 23) 
  

   cmd_list_reset()
   ##-- Generate Commands --##
   num_itr = math.ceil(n_head_per_hbm / (n_channel))
   for itr in range(num_itr):
     remainder = 0
     if (n_head_per_hbm / ((itr+1) * n_channel) < 1):
       remainder = n_head_per_hbm % n_channel
     key_addr = itr * partition_size 
     val_addr = key_addr + v_offset
     if remainder == 0:
       Attention(L, key_addr, val_addr, itr, n_channel, page_wise)
     else:
       Attention(L, key_addr, val_addr, itr, remainder, page_wise)


   ##-- Ovelapping Commands --##
   barrier = []
   for lch in range(n_channel):
     addr = lch * HBM_GS['ch']
     hex_addr = hex(addr)[2:]
     barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

   total_cmd = []
   for i in range(0, num_itr -1, 2):
     # Head0: Score
       ## WRGB
     total_cmd += cmd_score_wrgb[i]
       ## dummy MAC
     if i == 0:
       for j in range(valid_channels[i]):
         total_cmd.append(cmd_score_mac[i][0][j])
       ## BARRIER
     total_cmd += barrier

     length = math.ceil(L/n_pch/n_rank/n_bg/16)
     for j in range(0, length+1):
       ## MAC (Head0)
       if not j == length:
         stride = 16;
         for k in range(stride):
           if (j*stride+k) >= len(cmd_score_mac[i]):
             break;
           total_cmd += cmd_score_mac[i][j*stride+k]
       ## MVSB (Head0)
       if not j == 0:
         total_cmd += cmd_score_mvsb[i][j-1]
       ## WRGB (Head1)
       if not j == length:
         stride = int(n_bank*math.ceil(dhead /n_bank /n_mac)*math.ceil(valid_channels[i+1])/length);
         for k in range(stride):
           if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
             break;
           total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
       ## BARRIER
       if not j == length:
         total_cmd += barrier

     # Head0: SoftMax, Head1: Score
     length = math.ceil(L/n_pch/n_rank/n_bg/16)
     for j in range(0, length+1):
       ## MAC (Head1)
       if not j == length:
         stride = 16;
         for k in range(stride):
           if (j*stride+k) >= len(cmd_score_mac[i+1]):
             break;
           total_cmd += cmd_score_mac[i+1][j*stride+k]
       ## MVSB (Head1)
       if not j == 0:
         total_cmd += cmd_score_mvsb[i+1][j-1]
       ## SFM (Head0)
       if j == 0:
         total_cmd += cmd_sfm[i]
       ## MVGB (Head0)
       if not j == length:
         if j >= math.floor(length/2):
           stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2));
           for k in range(stride):
             if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i]):
               break;
             total_cmd.append(cmd_context_mvgb[i][(j-math.floor(length/2))*stride + k])
       ## BARRIER
       if not j == length:
         total_cmd += barrier

     # Head0: Context, Head1: Softmax
     length = math.ceil(dhead/n_bank/n_mac)
     for j in range(0, length+1):
       ## MAC (Head0)
       if not j == length:
         total_cmd += cmd_context_mac[i][j]
       ## MVSB (Head0)
       if not j == 0:
         total_cmd += cmd_context_mvsb[i][j-1]
       ## SFM (Head1)
       if j == 0:
         total_cmd += cmd_sfm[i+1]
       ## MVGB (Head1)
       if not j == length:
         if j >= math.floor(length/2):
           stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i+1])/math.ceil(length/2));
           for k in range(stride):
             if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i+1]):
               break;
             total_cmd.append(cmd_context_mvgb[i+1][(j-math.floor(length/2))*stride + k])
       ## BARRIER
       if not j == length:
         total_cmd += barrier

     # Head1: Context
     length = math.ceil(dhead/n_bank/n_mac)
     for j in range(0, length+1):
       ## MAC (Head0)
       if not j == length:
         total_cmd += cmd_context_mac[i][j]
       ## MVSB (Head0)
       if not j == 0:
         total_cmd += cmd_context_mvsb[i][j-1]
       ## BARRIER
       if not j == length:
         total_cmd += barrier


   if num_itr % 2 != 0:
     i = num_itr - 1

     # Score
       ## WRGB
     total_cmd += cmd_score_wrgb[i]
       ## BARRIER
     total_cmd += barrier

     length = math.ceil(L/n_pch/n_rank/n_bg/16)
     blk = n_pch * n_rank * n_bg * 16
     print(f"blk={blk}")

     for j in range(0, length+1):
       ## MAC
       if not j == length:
         stride = 16;
         for k in range(stride):
           if (j*stride+k) >= len(cmd_score_mac[i]):
             break;
           total_cmd += cmd_score_mac[i][j*stride+k]
       ## MVSB
       if not j == 0:
         # 报错前加上：
         print(f"L={L}, calc_len={length}, len_mvsb={len(cmd_score_mvsb[i])}, j={j}")
         assert j == 0 or (j-1) < len(cmd_score_mvsb[i]), "MVSB overflow"

         total_cmd += cmd_score_mvsb[i][j-1]
       ## BARRIER
       if not j == length:
         total_cmd += barrier

     # SoftMax
     ## SFM (Head0)
     total_cmd += cmd_sfm[i]
     ## MVGB (Head0)
     total_cmd += cmd_context_mvgb[i]
     ## BARRIER
     total_cmd += barrier

     # Context
     length = math.ceil(dhead/n_bank/n_mac)
     for j in range(0, length+1):
       ## MAC
       if not j == length:
         total_cmd += cmd_context_mac[i][j]
       ## MVSB
       if not j == 0:
         total_cmd += cmd_context_mvsb[i][j-1]
       ## BARRIER
       if not j == length:
         total_cmd += barrier


   trace_file = open(trace_file_name, 'w')
   for cmd in total_cmd:
     trace_file.write(cmd + "\n")

   trace_file.close()
"""


def run_attention(dhead, n_head_per_hbm, L, trace_file_name, page_wise=False, cluster_width=64, add_cluster=False):
    import math

    # ---- Helper ----
    def mac_iters(lst, stride=16):
        return math.ceil(len(lst) / stride) if lst else 0

    # ====== 基本参数 ======
    partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
    head_offset = partition_size
    v_offset = pow(2, 23)
    norm_offset = pow(2, 23)
    distance_offset = pow(2, 23)/(dhead * n_head_per_hbm)
    blk = n_pch * n_rank * n_bg * 16  # 你之前的 256，只是留作参考，不再用它控制循环
  
    # ====== 生成各阶段命令 ======
    cmd_list_reset()
    num_itr = math.ceil(n_head_per_hbm / n_channel)
    for itr in range(num_itr):
        remainder = 0
        if (n_head_per_hbm / ((itr + 1) * n_channel) < 1):
            remainder = n_head_per_hbm % n_channel
        key_addr = itr * partition_size
        val_addr = key_addr + v_offset
        # Our implementation: set address of norm and Centroid * Vector
        norm_addr = val_addr + norm_offset
        #key_new_addr = norm_addr + distance_offset
        #val_new_addr = key_new_addr + distance_offset + pow(2, 23)

        if remainder == 0:
            Attention(L, key_addr, val_addr, itr, n_channel, page_wise,itr, cluster_width=cluster_width, norm_addr=norm_addr, cluater_itr=16, add_cluster=add_cluster)
        else:
            Attention(L, key_addr, val_addr, itr, remainder, page_wise,itr, cluster_width=cluster_width, norm_addr=norm_addr, cluater_itr=16, add_cluster=add_cluster)
    
    # ====== Overlapping Commands ======
    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))
    #barrier = []
    total_cmd = []
    print(num_itr)

    if add_cluster:
      if page_wise:
        # Head0: norm
        for i in range(0, num_itr - 1, 2):
          total_cmd += barrier
          length = math.ceil(L/n_pch/n_rank/n_bg/16)
          print(length)
          for j in range(0, length+1):
            ## MAC (Head0)
            if not j == length:
              stride = 16;
              for k in range(stride):
                if (j*stride+k) >= len(cmd_norm_mac[i]):
                  break;
                total_cmd += cmd_norm_mac[i][j*stride+k]
            ## MVSB (Head0)
            if not j == 0:
              total_cmd += cmd_norm_mvsb[i][j-1]
            ## WRGB (Head1)
            if not j == length:
              stride = int(n_bank*math.ceil(dhead /n_bank /n_mac)*math.ceil(valid_channels[i+1])/length);
              for k in range(stride):
                if (j*stride+k) >= len(cmd_norm_mvgb[i+1]):
                  break;
                total_cmd.append(cmd_norm_mvgb[i+1][j*stride + k])
            ## BARRIER
            if not j == length:
              total_cmd += barrier

          # Head0: cos, Head1: norm
          length = math.ceil(L/n_pch/n_rank/n_bg/16)
          for j in range(0, length+1):
            ## MAC (Head1)
            if not j == length:
              stride = 16;
              for k in range(stride):
                if (j*stride+k) >= len(cmd_norm_mac[i+1]):
                  break;
                total_cmd += cmd_norm_mac[i+1][j*stride+k]
            ## MVSB (Head1)
            if not j == 0:
              total_cmd += cmd_norm_mvsb[i+1][j-1]
            ## MVGB (Head0)
            if not j == length:
              if j >= math.floor(length/2):
                stride = int(n_rank*n_bg*math.ceil(L*L/cluster_width/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2));
                for k in range(stride):
                  if ((j-math.floor(length/2))*stride + k) >= len(cmd_cos_mvgb[i]):
                    break;
                  total_cmd.append(cmd_cos_mvgb[i][(j-math.floor(length/2))*stride + k])
            ## BARRIER
            if not j == length:
              total_cmd += barrier

          # Head1: cos
          length = math.ceil(L*L/cluster_width/n_pch/n_rank/n_bg/16)
          for j in range(0, length+1):
            ## MAC (Head0)
            if not j == length:
              total_cmd += cmd_cos_mac[i][j]
            ## MVSB (Head0)
            if not j == 0:
              total_cmd += cmd_cos_mvgb[i][j-1]
            ## BARRIER
            if not j == length:
              total_cmd += barrier          
        if num_itr % 2 != 0:
          i = num_itr - 1
          total_cmd += cmd_norm_mvgb[i]
          # NORM1
          total_cmd += barrier
        
          length = math.ceil(L/n_pch/n_rank/n_bg/16)
          for j in range(0, length+1):
            ## MAC
            if not j == length:
              stride = 16;
              for k in range(stride):
                if (j*stride+k) >= len(cmd_norm_mac[i]):
                  break;
                total_cmd += cmd_norm_mac[i][j*stride+k]
            ## MVSB
            if not j == 0:
              total_cmd += cmd_norm_mvsb[i][j-1]
            ## BARRIER
            if not j == length:
              total_cmd += barrier
          ## MVGB (Head0)
          total_cmd += cmd_cos_mvgb[i]
          ## BARRIER
          total_cmd += barrier
          # Context
          length = math.ceil(L*L/cluster_width/n_bank/n_mac/16)
          for j in range(0, length+1):
            ## MAC
            if not j == length:
              total_cmd += cmd_cos_mac[i][j]
            ## MVSB
            if not j == 0:
              total_cmd += cmd_cos_mvsb[i][j-1]
            ## BARRIER
            if not j == length:
              total_cmd += barrier
      #print("&"*10)
      #print(len(cmd_norm_mvsb[2]))
      #print(len(cmd_norm_mac[2]))
      #print(len(cmd_norm_mvgb[2]))
      #print(len(cmd_cos_mvsb[2]))
      #print(len(cmd_cos_mac[2]))
      #print(len(cmd_cos_mvgb[2]))
      #print("&"*10)

      #exit()
    

    # -------- 处理两两 head 重叠的主体部分 --------
    for i in range(0, num_itr - 1, 2):
        # ---------------- Head0: Score 预处理 ----------------
        # WRGB
        total_cmd += cmd_score_wrgb[i]

        # dummy MAC
        
        if i == 0:
            for j in range(valid_channels[i]):
                total_cmd.append(cmd_score_mac[i][0][j])
        
        # BARRIER
        total_cmd += barrier

        # ======== Score 阶段：Head0 (MAC/MVSB) & Head1(WRGB) ========
        mac0_stride = 16
        mac0_iters = mac_iters(cmd_score_mac[i], mac0_stride)
        mvsb0_len = len(cmd_score_mvsb[i])

        # length 由真实块数驱动
        length = max(mac0_iters, mvsb0_len)

        for j in range(0, length + 1):
            # MAC (Head0)
            if j != length:
                for k in range(mac0_stride):
                    idx = j * mac0_stride + k
                    if idx >= len(cmd_score_mac[i]):
                        break
                    total_cmd += cmd_score_mac[i][idx]

            # MVSB (Head0)
            if j != 0 and (j - 1) < len(cmd_score_mvsb[i]):
                total_cmd += cmd_score_mvsb[i][j - 1]

            # WRGB (Head1)
            if j != length:
                wrgb1_stride = int(
                    n_bank
                    * math.ceil(dhead / n_bank / n_mac)
                    * math.ceil(valid_channels[i + 1])
                    / max(1, length)
                )
                for k in range(wrgb1_stride):
                    idx = j * wrgb1_stride + k
                    if idx >= len(cmd_score_wrgb[i + 1]):
                        break
                    total_cmd.append(cmd_score_wrgb[i + 1][idx])

            # BARRIER
            if j != length:
                total_cmd += barrier

        # ======== Head0: SoftMax, Head1: Score ========
        mac1_stride = 16
        mac1_iters = mac_iters(cmd_score_mac[i + 1], mac1_stride)
        mvsb1_len = len(cmd_score_mvsb[i + 1])
        mvgb0_len = len(cmd_context_mvgb[i])

        # 这里 length2 主要由 mac1 和 mvsb1 驱动；mvgb0 由于半程起步，后面再判断索引
        length2 = max(mac1_iters, mvsb1_len)

        for j in range(0, length2 + 1):
            # MAC (Head1)
            if j != length2:
                for k in range(mac1_stride):
                    idx = j * mac1_stride + k
                    if idx >= len(cmd_score_mac[i + 1]):
                        break
                    total_cmd += cmd_score_mac[i + 1][idx]

            # MVSB (Head1)
            if j != 0 and (j - 1) < len(cmd_score_mvsb[i + 1]):
                total_cmd += cmd_score_mvsb[i + 1][j - 1]

            # SFM (Head0)
            if j == 0:
                total_cmd += cmd_sfm[i]

            # MVGB (Head0) —— 从 length2//2 开始搬
            if j != length2 and j >= math.floor(length2 / 2):
                mvgb_stride0 = int(
                    n_rank
                    * n_bg
                    * math.ceil(L / (n_pch * n_rank * n_bg * n_mac))
                    * math.ceil(valid_channels[i])
                    / max(1, math.ceil(length2 / 2))
                )
                base = (j - math.floor(length2 / 2)) * mvgb_stride0
                for k in range(mvgb_stride0):
                    idx = base + k
                    if idx >= mvgb0_len:
                        break
                    total_cmd.append(cmd_context_mvgb[i][idx])

            # BARRIER
            if j != length2:
                total_cmd += barrier

        # ======== Head0: Context, Head1: Softmax ========
        ctx_mac0_len = len(cmd_context_mac[i])
        ctx_mvsb0_len = len(cmd_context_mvsb[i])
        mvgb1_len = len(cmd_context_mvgb[i + 1])
        # context 的 MAC / MVSB 是 1:1 对应（列表里本来就按 j 分）
        length3 = max(ctx_mac0_len, ctx_mvsb0_len)

        for j in range(0, length3 + 1):
            # MAC (Head0)
            if j != length3 and j < ctx_mac0_len:
                total_cmd += cmd_context_mac[i][j]

            # MVSB (Head0)
            if j != 0 and (j - 1) < ctx_mvsb0_len:
                total_cmd += cmd_context_mvsb[i][j - 1]

            # SFM (Head1)
            if j == 0:
                total_cmd += cmd_sfm[i + 1]

            # MVGB (Head1) —— 同样半程起步
            if j != length3 and j >= math.floor(length3 / 2):
                mvgb_stride1 = int(
                    n_rank
                    * n_bg
                    * math.ceil(L / (n_pch * n_rank * n_bg * n_mac))
                    * math.ceil(valid_channels[i + 1])
                    / max(1, math.ceil(length3 / 2))
                )
                base = (j - math.floor(length3 / 2)) * mvgb_stride1
                for k in range(mvgb_stride1):
                    idx = base + k
                    if idx >= mvgb1_len:
                        break
                    total_cmd.append(cmd_context_mvgb[i + 1][idx])

            # BARRIER
            if j != length3:
                total_cmd += barrier

        # ======== Head1: Context ========
        ctx_mac1_len = len(cmd_context_mac[i])
        ctx_mvsb1_len = len(cmd_context_mvsb[i])
        length4 = max(ctx_mac1_len, ctx_mvsb1_len)

        for j in range(0, length4 + 1):
            # MAC
            if j != length4 and j < ctx_mac1_len:
                total_cmd += cmd_context_mac[i][j]

            # MVSB
            if j != 0 and (j - 1) < ctx_mvsb1_len:
                total_cmd += cmd_context_mvsb[i][j - 1]

            # BARRIER
            if j != length4:
                total_cmd += barrier

    # -------- 奇数个 head 的最后一个尾巴 --------
    if num_itr % 2 != 0:
        i = num_itr - 1

        # Score
        # WRGB
        total_cmd += cmd_score_wrgb[i]
   
        # BARRIER
        total_cmd += barrier

        mac_tail_stride = 16
        mac_tail_iters = mac_iters(cmd_score_mac[i], mac_tail_stride)
        mvsb_tail_len = len(cmd_score_mvsb[i])
        length_tail = max(mac_tail_iters, mvsb_tail_len)

        for j in range(0, length_tail + 1):
            # MAC
            if j != length_tail:
                for k in range(mac_tail_stride):
                    idx = j * mac_tail_stride + k
                    if idx >= len(cmd_score_mac[i]):
                        break
                    total_cmd += cmd_score_mac[i][idx]

            # MVSB
            if j != 0 and (j - 1) < len(cmd_score_mvsb[i]):
                # print(f"L={L}, calc_len={length_tail}, len_mvsb={len(cmd_score_mvsb[i])}, j={j}")
                # assert j == 0 or (j-1) < len(cmd_score_mvsb[i]), "MVSB overflow"
                total_cmd += cmd_score_mvsb[i][j - 1]

            # BARRIER
            if j != length_tail:
                total_cmd += barrier

        # SoftMax
        total_cmd += cmd_sfm[i]
        total_cmd += cmd_context_mvgb[i]
        total_cmd += barrier

        # Context
        ctx_mac_tail_len = len(cmd_context_mac[i])
        ctx_mvsb_tail_len = len(cmd_context_mvsb[i])
        length_ctx_tail = max(ctx_mac_tail_len, ctx_mvsb_tail_len)

        for j in range(0, length_ctx_tail + 1):
            # MAC
            if j != length_ctx_tail and j < ctx_mac_tail_len:
                total_cmd += cmd_context_mac[i][j]

            # MVSB
            if j != 0 and (j - 1) < ctx_mvsb_tail_len:
                total_cmd += cmd_context_mvsb[i][j - 1]

            # BARRIER
            if j != length_ctx_tail:
                total_cmd += barrier
    print(len(total_cmd))
    #exit()
    # ====== 写入 trace 文件 ======
    with open(trace_file_name, 'w') as trace_file:
        for cmd in total_cmd:
            trace_file.write(cmd + "\n")


def main():
  global dhead, max_L, data_size, n_mac

  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=64,
                      help="Number of heads, default=64")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=409600, 
                      help="maximum L, default= 409600")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace", 
                      help="output path")
  parser.add_argument("-pw", "--page_wise", action="store_true", 
                      help="enable page-wise attention (use 1/4 of pages)")
  parser.add_argument("-ps", "--page_size", type=int, default=16,
                      help="tokens per page for page-wise attention (default: 16)")
  parser.add_argument("-psr", "--page_select_ratio", type=float, default=0.25,
                      help="ratio of pages to use for each token (default: 0.25)")
  parser.add_argument("-v", "--verbose", action="store_true",
                       help="show detailed page layout information")
  parser.add_argument("--kv_budget", type=int, default=1024,
                    help="固定 KV budget (pair)，0 表示按比例 page_wise")
  parser.add_argument("-dbug", action="store_true")
  parser.add_argument("--add_cluster", action="store_true",
                       help="whether to add cluster overhead")
  

  args = parser.parse_args()
  debug = args.dbug
  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead 

  data_size = args.dbyte
  n_mac = int(HBM_GS['col'] / data_size)

  # Update global page-wise configuration
  if args.page_wise:
    global PAGE_SIZE, PAGE_SELECT_RATIO, KV_BUDGET
    PAGE_SIZE = args.page_size
    PAGE_SELECT_RATIO = args.page_select_ratio
    KV_BUDGET = args.kv_budget          # 若为 0 则退化到比例模式

  print("------   Make a trace of bank-level AttAcc   ------")
  
  # Calculate page configuration
  total_pages = math.ceil(L / PAGE_SIZE) if args.page_wise else 0
  if args.page_wise:
    if args.kv_budget > 0:
        _, active_pages = calculate_page_config_fixed_budget(L, args.kv_budget)
    else:
        _, active_pages = calculate_page_config(L)

  
  if args.page_wise:
    print(f" Page-wise attention mode enabled")
    #print_page_statistics(L, total_pages, active_pages)

  print(f" Configuration Arguments:")
  args_dict = vars(args)
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")
  run_attention(dhead, n_head_per_hbm, L, args.output, args.page_wise, cluster_width=64, add_cluster=args.add_cluster)



if __name__ == "__main__":
  main()
