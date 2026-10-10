#!/usr/bin/env python3
"""Bind selected compiler footprints to independently decoded finite observations.

No timing rules are defined here. Static streams come from the shared resolver;
DMA completion is anchored to actual matching WAIT acceptance, never a latency.
"""
import hashlib
import struct

class BindingError(ValueError):
    pass

def require(condition,message):
    if not condition: raise BindingError(message)

def integer(value,label):
    require(type(value) is int and value>=0,'invalid '+label)
    return value

def finite_stream(stream,issue):
    require(stream.get('anywhere') is False and stream.get('at_completion') is False,
            'concrete issue-relative access stream required')
    first=integer(stream['first'],'access first');count=integer(stream['count'],'access count')
    age=integer(stream['age'],'access age');step=integer(stream['step'],'access step')
    require(0<count<=256 and step>0,'unsupported finite access count/step')
    require(type(stream['write']) is bool,'invalid access direction')
    return [(stream['resource'],stream['write'],first+n,issue+age+n*step) for n in range(count)]

def decoded_operands(word):
    opcode=word&127;rd=(word>>7)&31;rs1=(word>>15)&31;rs2=(word>>20)&31
    result={'rd':0,'rs1':0,'rs2':0,'immediate':0,'release':False}
    immediate=word>>20; signed=immediate-4096 if immediate&2048 else immediate
    if opcode==0x13 and (word>>12)&7==0:
        mnemonic='addi';result.update(rd=rd,rs1=rs1,immediate=signed)
    elif opcode==0x37:
        mnemonic='lui';result.update(rd=rd,immediate=word>>12)
    elif opcode==0x67 and (word>>12)&7==1:
        mnemonic='delay';result.update(immediate=immediate)
    elif opcode==0x07:
        mnemonic='vstore' if word&(1<<13) else 'vload'
        result.update(rd=(word>>7)&63,rs1=rs1,immediate=signed)
    elif opcode==0x6b:
        mnemonic='vtrpose.xlu';result.update(rd=(word>>7)&63,rs1=(word>>13)&63)
    elif opcode==0x57 and word>>25==3:
        mnemonic='vmul.bf16';result.update(rd=(word>>7)&63,rs1=(word>>13)&63,rs2=(word>>19)&63)
    elif opcode==0x7b and word>>25 in (0,1):
        mnemonic='dma.'+('store' if word>>25 else 'load')+'.ch'+str((word>>12)&7)
        result.update(rd=rd,rs1=rs1,rs2=rs2)
    elif opcode==0x7f and word>>25 in (0,1):
        mnemonic='dma.'+('wait' if word>>25 else 'config')+'.ch'+str((word>>12)&7)
        if not word>>25: result.update(rs1=rs1)
    elif opcode==0x73 and (word>>12)&7==1:
        mnemonic='csrrw';result.update(rd=rd,rs1=rs1,immediate=immediate)
    elif word==0x73:
        mnemonic='ecall'
    else: raise BindingError('unsupported word in finite export binding')
    return mnemonic,result

def bind_export(export,observed,evidence,words,resolver_id):
    require(export.get('schema') in ('atlas.resolved_rtl_timing.v0','atlas.resolved_rtl_timing.v1') and export.get('target_config')=='EE290SimConfig','export schema/target mismatch')
    require(export.get('qualification')=='conditional' and export.get('scheduling_qualified') is False,'qualification promotion rejected')
    require(export.get('resolver')=={'id':resolver_id,'version':1},'selected resolver mismatch')
    require(export.get('evidence')==evidence,'selected evidence identity mismatch')
    require(all(type(word) is int and 0<=word<2**32 for word in words),'invalid selected program words')
    digest=hashlib.sha256(b''.join(struct.pack('<I',word) for word in words)).hexdigest()
    require(export.get('program',{}).get('words')==words and export['program'].get('word_count')==len(words) and export['program'].get('words_sha256')==digest,'export differs from selected emitted words')
    require(observed.get('target_config')=='EE290SimConfig','observed target mismatch')
    require(observed.get('data_encoding')=='little_endian_bytes_hex','unsupported observed data encoding')
    require(export['program'].get('hash_encoding')=='little_endian_u32','unsupported program hash encoding')
    instructions=export['instructions'];require(len(instructions)==len(words),'incomplete resolved instruction export')
    retired={item['word_index']:item for item in observed['instructions']}
    require(len(retired)==len(observed['instructions']),'duplicate observed instruction index')
    halts=[item for item in observed['endpoints'] if item['kind']=='halt']
    require(len(halts)==1,'one terminal acceptance required')
    halt=halts[0];retired[halt['word_index']]=halt
    require(set(retired)==set(range(len(words))),'incomplete actual retirement/terminal words')
    dynamic=export['schema'].endswith('.v1');origins={};epoch=0
    first_edge=retired[0]['edge'];waits={item['word_index']:item for item in observed['endpoints'] if item['kind']=='wait'}
    for index,item in enumerate(instructions):
        require(item['word_index']==index and item['word_u32']==words[index] and retired[index]['word_u32']==words[index],'resolved/observed instruction word mismatch')
        mnemonic,operands=decoded_operands(words[index])
        require(item['mnemonic']==mnemonic and item['operands']==operands,'exported finite instruction metadata differs from selected word')
        edge=integer(retired[index]['edge'],'observed issue edge')
        if index: require(edge>retired[index-1]['edge'],'actual instruction edges do not advance')
        if dynamic:
            current=integer(item['issue_epoch'],'issue epoch');offset=integer(item['epoch_offset'],'epoch offset')
            lower=integer(item['minimum_issue_cycle'],'minimum issue cycle')
            require(edge-first_edge>=lower,'observed issue precedes exported lower bound')
            if index in waits:
                require(current==epoch+1 and offset==0 and item.get('event_kind')=='matching_wait_acceptance','matching WAIT does not start the next epoch')
                epoch=current;origins[epoch]=edge
            else: require(current==epoch,'epoch changes without matching WAIT acceptance')
            if current not in origins:
                require(current==0 and index==0 and offset==0,'unsupported epoch origin')
                origins[current]=edge
            require(edge==origins[current]+offset,'observed intra-epoch issue differs from exported offset')
        else:
            require(not waits,'static export cannot describe dynamic WAIT acceptance')
            cycle=integer(item['logical_issue_cycle'],'logical issue cycle')
            if index==0: origins[0]=edge-cycle
            require(edge==origins[0]+cycle,'observed issue differs from exported timeline')
        if index==len(words)-1:
            require(item.get('event_kind')=='terminal_acceptance' and item['mnemonic']=='ecall','terminal acceptance convention mismatch')
    commands={command['word_index']:command for command in observed['commands']}
    require(len(commands)==len(observed['commands']),'duplicate observed command index')
    resolved_commands=[];accesses_checked=0;dynamic_intervals=[]
    for index,item in enumerate(instructions):
        mnemonic=item['mnemonic'];word=words[index];operands=item['operands']
        is_command=mnemonic in ('vload','vstore','vtrpose.xlu','vmul.bf16') or mnemonic.startswith(('dma.load.ch','dma.store.ch'))
        if not is_command: continue
        require(index in commands,'exported compute command absent from observations')
        command=commands[index];require(command['word_u32']==word and command['edge']==retired[index]['edge'],'command differs from actual selected instruction')
        kind=command['engine'];footprint=item['footprint'];issue=command['edge']
        if kind=='vls':
            require(mnemonic==('vload' if command['op']=='load' else 'vstore') and operands['rd']==command['mreg'],'resolved VLS operation/register mismatch')
            observed_points=[]
            for field,write in [('reads',False),('writes',True)]:
                resource='vmem' if (command['op']=='load')!=(write) else 'mreg'
                for event in command[field]:
                    address=event['id'] if resource=='vmem' else 32*event['id']+event['row']
                    observed_points.append((resource,write,address,event['edge']))
        elif kind=='xlu':
            require(mnemonic=='vtrpose.xlu' and operands['rd']==command['dst'] and operands['rs1']==command['src'],'resolved XLU operands mismatch')
            observed_points=[('mreg',write,32*e['id']+e['row'],e['edge']) for field,write in [('reads',False),('writes',True)] for e in command[field]]
        elif kind=='vmul':
            require(mnemonic=='vmul.bf16' and (operands['rd'],operands['rs1'],operands['rs2'])==(command['dst'],command['lhs'],command['rhs']),'resolved VMUL operands mismatch')
            for field in ('reads','responses','writes'):
                bases=(command['dst'],) if field=='writes' else (command['lhs'],command['rhs'])
                for event in command[field]:
                    port=event.get('port')
                    require(type(port) is int and 0<=port<len(bases) and event['id']-bases[port] in (0,1),'VMUL event lacks its explicit pair port')
            observed_points=[('mreg',write,32*e['id']+e['row'],e['edge']) for field,write in [('reads',False),('writes',True)] for e in command[field]]
        elif kind=='dma':
            channel=integer(command['channel'],'DMA channel');direction=command['op']
            require(mnemonic==f'dma.{direction}.ch{channel}','resolved DMA operation/channel mismatch')
            require(dynamic and footprint.get('dma_async') is True and footprint.get('completion')=='matching_dma_wait' and all(footprint.get(k) is None for k in ('done_age','dma_cycles','read_release','write_release')),'DMA completion was assigned a fixed latency')
            memory=[a for a in footprint['accesses'] if a['resource']=='vmem'];dram=[a for a in footprint['accesses'] if a['resource']=='dram']
            require(len(memory)==len(dram)==1,'explicit DMA VMEM and conservative DRAM effects required')
            stream=memory[0]
            require(stream.get('at_completion') is True and stream.get('anywhere') is False and stream.get('age') is None and stream.get('step') is None and stream.get('lifetime')=='launch_through_matching_wait','DMA VMEM effect lacks dynamic completion lifetime')
            require(dram[0].get('at_completion') is True and dram[0].get('anywhere') is True and dram[0].get('age') is None and dram[0].get('step') is None,'DMA DRAM effect must remain dynamically conservative')
            require(stream['write']==(direction=='load') and dram[0]['write']==(direction=='store'),'DMA effect direction mismatch')
            require(stream['first']==command['vmem_word']//8 and stream['count']==command['size']//32,'resolved DMA captured VMEM range mismatch')
            memory_events=command['memory']
            require([e['line'] for e in memory_events]==list(range(stream['first'],stream['first']+stream['count'])),'observed DMA VMEM lines differ from exported range')
            matches=[e for e in waits.values() if e['channel']==channel and e['edge']==command['release_edge']]
            require(len(matches)==1 and all(issue<=e['edge']<matches[0]['edge'] for e in memory_events+command['requests']+command['responses']),'DMA effects do not drain before actual matching WAIT')
            require(matches[0]['word_index']>index,'matching WAIT precedes selected launch')
            dynamic_intervals.append({'word_index':index,'channel':channel,'launch_edge':issue,'matching_wait_edge':matches[0]['edge'],'observed_elapsed_edges':matches[0]['edge']-issue,'compiler_latency':None})
            resolved_commands.append(index);accesses_checked+=len(memory_events);continue
        else: raise BindingError('unsupported observed command engine')
        exported_points=[]
        for stream in footprint['accesses']:
            if stream['resource'] in ('mreg','vmem'): exported_points.extend(finite_stream(stream,issue))
        require(sorted(exported_points)==sorted(observed_points),'observed access resource/address/age differs from shared resolver footprint')
        reads={(event.get('port',0),event['id'],event['row']):event for event in command['reads']}
        require(len(reads)==len(command['reads']),'duplicate command read row')
        require(len(command['responses'])==len(reads),'incomplete source response stream')
        seen_responses=set()
        for response in command['responses']:
            key=(response.get('port',0),response['id'],response['row'])
            require(key not in seen_responses,'duplicate command response row')
            seen_responses.add(key)
            require(key in reads and response['edge']==reads[key]['edge']+1 and response['data_hex']==reads[key]['data_hex'],'read response differs from observed one-cycle source stream')
        inclusive=[integer(footprint[k],k) for k in ('done_age','read_release','write_release')]
        inclusive.extend(integer(hold['to'],'hold inclusive end') for hold in footprint['holds'])
        require(command['release_edge']==issue+max(inclusive)+1,'observed engine release differs from inclusive resolver holds')
        # Exported lifetimes must cover every observed MREG request/response/write.
        age=lambda event:event['edge']-issue
        sources=command['reads']+command['responses'] if kind!='vls' or command['op']=='store' else []
        destinations=command['writes'] if kind!='vls' or command['op']=='load' else []
        require(sorted(footprint['mreg_reads'])==sorted({e['id'] for e in sources}) and sorted(footprint['mreg_writes'])==sorted({e['id'] for e in destinations}),'exported MREG lifetime registers differ from observed streams')
        require(footprint['read_release']>=max(map(age,sources),default=0) and footprint['write_release']>=max(map(age,destinations),default=0) and footprint['done_age']>=max(map(age,command['reads']+command['responses']+command['writes'])),'exported lifetime ends before an observed access')
        for unit in {'vls':(),'xlu':('XLU',),'vmul':('VPU',)}[kind]+('VLOAD path','VSTORE path'):
            require(sum(h['unit']==unit and h['index']==0 and h['from']==0 and h['to']==footprint['done_age'] for h in footprint['holds'])==1,'serialized '+unit+' hold absent or shortened')
        resolved_commands.append(index);accesses_checked+=len(observed_points)
    require(set(resolved_commands)==set(commands),'observed command lacks selected exported footprint')
    return {'schema':'atlas.ee290_compute_export_binding.v0','target_config':'EE290SimConfig','scheduling_qualified':False,'resolver':resolver_id,'instruction_words_bound':len(words),'memory_access_elements_bound':accesses_checked,'epoch_origins':[{'epoch':key,'actual_edge':value} for key,value in sorted(origins.items())],'dynamic_dma_intervals':dynamic_intervals,'release_coverage':'observed combined engine release equals maximum inclusive exported hold; serialized VLOAD/VSTORE/engine holds span done_age; MREG read/write/done lifetimes bound every observed access; conservative excess is not measured','scope':'finite selected instruction/access/combined release and matching-WAIT epoch correspondence'}
